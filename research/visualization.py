#!/usr/bin/env python3
"""
Robust Visualization Module for Malware Classification System
Handles Grad-CAM generation and heatmap visualization with comprehensive error handling
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import logging
import traceback
from data_handler import DataHandler
from model_builder import CastLayer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Visualization")


class GradCAMGenerator:
    def __init__(self, model, img_size=256, output_dir=None):
        """
        Initialize the Grad-CAM generator

        Args:
            model: The trained TensorFlow model
            img_size: Size of the input images
            output_dir: Directory to save Grad-CAM visualizations
        """
        self.model = model
        self.img_size = img_size
        self.output_dir = output_dir
        self.conv_layers = []

        try:
            if output_dir:
                os.makedirs(os.path.join(output_dir, 'gradcam'), exist_ok=True)

            # Get list of convolutional layers from the model
            if model is not None:
                self.conv_layers = [layer.name for layer in model.layers
                                    if isinstance(layer, tf.keras.layers.Conv2D)]
                logger.info(f"Found {len(self.conv_layers)} convolutional layers")
            else:
                logger.warning("No model provided, GradCAM functionality will be limited")
        except Exception as e:
            logger.error(f"Error initializing GradCAMGenerator: {e}")
            logger.error(traceback.format_exc())

    def compute_gradcam(self, img, layer_name):
        """
        Compute Grad-CAM heatmap for a single image

        Args:
            img: Input image as a NumPy array
            layer_name: Name of the layer to use for Grad-CAM

        Returns:
            Grad-CAM heatmap as a NumPy array
        """
        try:
            if self.model is None:
                logger.warning("No model available for GradCAM computation")
                return np.zeros((10, 10))  # Return empty heatmap

            # Check if layer exists in the model
            try:
                self.model.get_layer(layer_name)
            except ValueError:
                logger.warning(f"Layer '{layer_name}' not found in the model")
                if self.conv_layers:
                    layer_name = self.conv_layers[-1]
                    logger.info(f"Using last convolutional layer instead: {layer_name}")
                else:
                    logger.error("No convolutional layers found in the model")
                    return np.zeros((10, 10))

            # Create the Grad-CAM model with outputs from the specified conv layer and final output
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[self.model.get_layer(layer_name).output, self.model.output]
            )

            # Prepare image: convert and add batch dimension
            img = tf.convert_to_tensor(img, dtype=tf.float32)
            img_batch = tf.expand_dims(img, axis=0)

            # Record operations for automatic differentiation
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_batch)

                # Handle different model output formats
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # For multi-class models, use the most confident class
                    class_idx = tf.argmax(predictions[0])
                    loss = predictions[0, class_idx]
                else:
                    # For binary classifiers with a single sigmoid output
                    loss = predictions[0, 0]

            # Compute gradients of the loss with respect to the conv layer output
            grads = tape.gradient(loss, conv_output)
            # Pool the gradients over the spatial dimensions
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # Multiply each feature map in the conv output by the pooled gradients
            cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
            # Apply ReLU so that only positive contributions remain
            cam = tf.nn.relu(cam)
            # Normalize the heatmap to the range [0, 1]
            max_val = tf.reduce_max(cam)
            if max_val > 0:
                cam = cam / (max_val + tf.keras.backend.epsilon())

            return cam.numpy()

        except Exception as e:
            logger.error(f"Error computing GradCAM: {e}")
            logger.error(traceback.format_exc())
            return np.zeros((10, 10))  # Return empty heatmap on error

    def generate_heatmaps(self, X_data, y_data, layer_name=None, num_classes=2):
        """
        Generate Grad-CAM heatmaps for a set of images and compute average heatmaps.

        Args:
            X_data: Input images as a NumPy array.
            y_data: Labels for the input images.
            layer_name: Name of the layer to use for Grad-CAM (if None, use the last conv layer).
            num_classes: Number of classes (default is 2).

        Returns:
            Dictionary containing average heatmaps for each class and overall.
        """
        try:
            if not self.conv_layers:
                logger.warning("No convolutional layers found in the model")
                return {}

            if layer_name is None:
                # Use the last convolutional layer by default
                layer_name = self.conv_layers[-1]

            logger.info(f"Generating Grad-CAM heatmaps for layer: {layer_name}")

            # Predict class probabilities using the model
            try:
                y_pred_prob = self.model.predict(X_data)

                # Handle different model output formats
                if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
                    # For multi-class, get the predicted class with highest probability
                    y_pred = np.argmax(y_pred_prob, axis=1)
                else:
                    # For binary classification with sigmoid output
                    y_pred_prob = y_pred_prob.flatten()
                    y_pred = (y_pred_prob > 0.5).astype(int)
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return {}

            # Initialize lists to store heatmaps for each class
            class_heatmaps = [[] for _ in range(num_classes)]
            correct_class_heatmaps = [[] for _ in range(num_classes)]

            # Set a limit on the number of samples to process to avoid excessive computation
            max_samples = min(len(X_data), 100)  # Limit to 100 samples

            # Compute Grad-CAM for each sample and group by true label
            for i in range(max_samples):
                try:
                    # Ensure the class index is within the valid range
                    class_idx = int(y_data[i]) % num_classes

                    # Compute GradCAM for this sample
                    cam = self.compute_gradcam(X_data[i], layer_name)

                    # Skip if CAM computation failed (returned empty array)
                    if cam.size == 0 or np.all(cam == 0):
                        continue

                    class_heatmaps[class_idx].append(cam)
                    if y_pred[i] == y_data[i]:
                        correct_class_heatmaps[class_idx].append(cam)
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue

            # Compute average heatmaps for each class and overall
            avg_heatmaps = {}
            for class_idx in range(num_classes):
                if class_heatmaps[class_idx]:
                    avg_heatmaps[f'class_{class_idx}'] = np.mean(class_heatmaps[class_idx], axis=0)
                if correct_class_heatmaps[class_idx]:
                    avg_heatmaps[f'correct_class_{class_idx}'] = np.mean(correct_class_heatmaps[class_idx], axis=0)

            all_heatmaps = []
            for class_list in class_heatmaps:
                all_heatmaps.extend(class_list)

            if all_heatmaps:
                avg_heatmaps['overall'] = np.mean(all_heatmaps, axis=0)

            # Save average heatmaps if an output directory is provided
            if self.output_dir:
                for name, heatmap in avg_heatmaps.items():
                    try:
                        np.save(
                            os.path.join(self.output_dir, 'gradcam', f'{name}_{layer_name}_heatmap.npy'),
                            heatmap
                        )
                    except Exception as e:
                        logger.warning(f"Error saving heatmap {name}: {e}")

            return avg_heatmaps

        except Exception as e:
            logger.error(f"Error generating heatmaps: {e}")
            logger.error(traceback.format_exc())
            return {}

    def visualize_heatmaps(self, heatmaps, layer_name=None, class_names=None):
        """
        Visualize Grad-CAM heatmaps

        Args:
            heatmaps: Dictionary of heatmaps from generate_heatmaps.
            layer_name: Name of the layer used for Grad-CAM.
            class_names: Names of the classes (default: ['Class 0', 'Class 1']).
        """
        try:
            if not heatmaps:
                logger.warning("No heatmaps to visualize")
                return

            if layer_name is None:
                layer_name = 'conv'

            if class_names is None:
                class_names = ['Class 0', 'Class 1']

            for name, heatmap in heatmaps.items():
                try:
                    # Resize heatmap to the standard image size
                    heatmap_resized = cv2.resize(heatmap, (self.img_size, self.img_size))

                    plt.figure(figsize=(10, 8))
                    plt.imshow(heatmap_resized, cmap='jet')
                    plt.colorbar(label='Activation')

                    # Determine title based on key names
                    if 'class_0' in name and len(class_names) > 0:
                        title = f'Average Heatmap for {class_names[0]}'
                    elif 'class_1' in name and len(class_names) > 1:
                        title = f'Average Heatmap for {class_names[1]}'
                    elif 'correct_class_0' in name and len(class_names) > 0:
                        title = f'Avg Heatmap for Correctly Classified {class_names[0]}'
                    elif 'correct_class_1' in name and len(class_names) > 1:
                        title = f'Avg Heatmap for Correctly Classified {class_names[1]}'
                    elif 'overall' in name:
                        title = 'Overall Average Heatmap'
                    else:
                        title = name

                    plt.title(title)
                    plt.axis('off')
                    plt.tight_layout()

                    if self.output_dir:
                        save_path = os.path.join(self.output_dir, 'gradcam', f'{name}_{layer_name}_heatmap.png')
                        plt.savefig(save_path, dpi=300)
                        logger.info(f"Saved heatmap visualization to {save_path}")

                    plt.close()
                except Exception as e:
                    logger.warning(f"Error visualizing heatmap {name}: {e}")
                    plt.close()
        except Exception as e:
            logger.error(f"Error in visualize_heatmaps: {e}")
            plt.close('all')  # Ensure all figures are closed

    def create_sample_overlays(self, X_samples, y_samples, layer_name=None, num_samples=5, class_names=None):
        """
        Create overlays of Grad-CAM heatmaps on a few sample images per class

        Args:
            X_samples: Sample images as a NumPy array.
            y_samples: Labels for the sample images.
            layer_name: Name of the layer to use for Grad-CAM.
            num_samples: Number of samples per class to visualize.
            class_names: Names of the classes (default: ['Benign', 'Malware']).
        """
        try:
            if not self.conv_layers:
                logger.warning("No convolutional layers found in the model")
                return

            if layer_name is None:
                layer_name = self.conv_layers[-1]

            if class_names is None:
                class_names = ['Benign', 'Malware']

            # Initialize overlay_dir with a default value of None
            overlay_dir = None

            if self.output_dir:
                try:
                    overlay_dir = os.path.join(self.output_dir, 'gradcam', 'overlays')
                    os.makedirs(overlay_dir, exist_ok=True)
                    logger.info(f"Created overlay directory at {overlay_dir}")
                except Exception as e:
                    logger.warning(f"Error creating overlay directory: {e}")
                    # Keep overlay_dir as None if directory creation fails

            # Group sample indices by class
            class_samples = {}
            for i in range(len(X_samples)):
                try:
                    class_idx = int(y_samples[i])
                    if class_idx not in class_samples:
                        class_samples[class_idx] = []
                    class_samples[class_idx].append(i)
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue

            # Process each class (limit to num_samples per class)
            for class_idx, sample_indices in class_samples.items():
                try:
                    # Skip if class_idx is out of range for class_names
                    if class_idx >= len(class_names):
                        logger.warning(f"Class index {class_idx} out of range for class_names")
                        continue

                    # Limit samples to process
                    sample_indices = sample_indices[:min(num_samples, len(sample_indices))]

                    for i, idx in enumerate(sample_indices):
                        try:
                            img = X_samples[idx]
                            cam = self.compute_gradcam(img, layer_name)

                            # Skip if CAM computation failed
                            if cam.size == 0 or np.all(cam == 0):
                                continue

                            plt.figure(figsize=(12, 5))
                            try:
                                self._create_overlay_subplot(plt, 0, img, cam,
                                                             f"{class_names[class_idx]} - Sample {i + 1}")
                            except Exception as e:
                                logger.warning(f"Error creating overlay subplot: {e}")
                                plt.close()
                                continue

                            plt.tight_layout()

                            # Only try to save if overlay_dir is defined
                            if overlay_dir is not None:
                                try:
                                    save_path = os.path.join(
                                        overlay_dir,
                                        f'{class_names[class_idx].lower()}_sample_{i + 1}_{layer_name}.png'
                                    )
                                    plt.savefig(save_path, dpi=300)
                                    logger.info(f"Saved sample overlay to {save_path}")
                                except Exception as e:
                                    logger.warning(f"Error saving figure: {e}")

                            plt.close()
                        except Exception as e:
                            logger.warning(f"Error processing sample {idx}: {e}")
                            plt.close()
                except Exception as e:
                    logger.warning(f"Error processing class {class_idx}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in create_sample_overlays: {e}")
            plt.close('all')  # Ensure all figures are closed

    def _create_overlay_subplot(self, plt_obj, idx, img, cam, title):
        """
        Create a subplot with the original image and its Grad-CAM overlay

        Args:
            plt_obj: Matplotlib pyplot instance.
            idx: Subplot index.
            img: Original image.
            cam: Grad-CAM heatmap.
            title: Title for the subplot.
        """
        try:
            # Determine image dimensions based on shape
            if len(img.shape) == 3:
                img_height, img_width = img.shape[:2]
            else:
                img_height, img_width = img.shape

            # Resize CAM to match image dimensions
            cam_resized = cv2.resize(cam, (img_width, img_height))

            # Convert to color heatmap
            heatmap = np.uint8(255 * cam_resized)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Convert BGR to RGB if needed
            if heatmap.shape[-1] == 3:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Convert image to RGB for overlay
            if len(img.shape) == 3 and img.shape[-1] == 1:
                # Single channel image
                img_rgb = np.tile(np.uint8(img * 255), (1, 1, 3))
            elif len(img.shape) == 2:
                # Grayscale image
                img_rgb = np.stack([np.uint8(img * 255)] * 3, axis=-1)
            else:
                # Already RGB or similar
                img_rgb = np.uint8(img * 255)

            # Create overlay
            overlay = heatmap * 0.4 + img_rgb * 0.6
            overlay = np.clip(overlay, 0, 255).astype('uint8')

            # Plot original image
            plt_obj.subplot(1, 2, 1)
            if len(img.shape) == 3 and img.shape[-1] == 1:
                plt_obj.imshow(img[:, :, 0], cmap='gray')
            else:
                plt_obj.imshow(img)
            plt_obj.title(f"{title} - Original")
            plt_obj.axis('off')

            # Plot overlay
            plt_obj.subplot(1, 2, 2)
            plt_obj.imshow(overlay)
            plt_obj.title(f"{title} - Grad-CAM Overlay")
            plt_obj.axis('off')
        except Exception as e:
            logger.warning(f"Error in _create_overlay_subplot: {e}")
            raise  # Re-raise to handle in the calling function

    def create_average_overlays(self, X_data, y_data, avg_heatmaps, class_names=None):
        """
        Create overlays using the average image and the averaged heatmap for each class,
        computed over all samples in the class.

        Args:
            X_data: Input images as a NumPy array.
            y_data: Corresponding labels.
            avg_heatmaps: Dictionary of average heatmaps as produced by generate_heatmaps.
            class_names: List of class names (default: ['Benign', 'Malware']).
        """
        try:
            if class_names is None:
                class_names = ['Benign', 'Malware']

            if not avg_heatmaps:
                logger.warning("No average heatmaps provided")
                return

            # For each class, compute the average image across all samples
            try:
                classes = np.unique(y_data)
            except Exception as e:
                logger.warning(f"Error getting unique classes: {e}")
                return

            for class_idx in classes:
                try:
                    # Skip if class index is out of range for class_names
                    if class_idx >= len(class_names):
                        logger.warning(f"Class index {class_idx} out of range for class_names")
                        continue

                    # Collect images for this class
                    class_images = []
                    for i in range(len(X_data)):
                        if int(y_data[i]) == class_idx:
                            class_images.append(X_data[i])

                    if not class_images:
                        logger.warning(f"No images found for class {class_idx}")
                        continue

                    # Compute average image
                    avg_image = np.mean(class_images, axis=0)

                    # Retrieve the corresponding average heatmap
                    heatmap_key = f'class_{class_idx}'
                    if heatmap_key not in avg_heatmaps:
                        logger.warning(f"No average heatmap found for class {class_idx}")
                        continue

                    heatmap = avg_heatmaps[heatmap_key]

                    # Resize heatmap to match the average image dimensions
                    if len(avg_image.shape) == 3:
                        height, width = avg_image.shape[:2]
                    else:
                        height, width = avg_image.shape

                    heatmap_resized = cv2.resize(heatmap, (width, height))

                    # Convert heatmap to color
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                    # Convert average image to uint8 (assuming it is in [0,1] range)
                    avg_image_uint8 = np.uint8(avg_image * 255)
                    if len(avg_image.shape) == 2:
                        avg_image_uint8 = cv2.cvtColor(avg_image_uint8, cv2.COLOR_GRAY2BGR)

                    # Create overlay from the average image and averaged heatmap
                    overlay = cv2.addWeighted(colored_heatmap, 0.4, avg_image_uint8, 0.6, 0)
                    overlay = np.clip(overlay, 0, 255).astype('uint8')

                    # Create side-by-side visualization
                    if self.output_dir:
                        try:
                            overlay_path = os.path.join(
                                self.output_dir,
                                'gradcam',
                                f'avg_overlay_class_{class_idx}.png'
                            )

                            plt.figure(figsize=(12, 6))
                            # Left subplot: average sample image
                            plt.subplot(1, 2, 1)
                            # Display as grayscale if the image is single channel
                            if len(avg_image.shape) == 2 or (len(avg_image.shape) == 3 and avg_image.shape[-1] == 1):
                                plt.imshow(avg_image.squeeze(), cmap='gray')
                            else:
                                plt.imshow(avg_image)
                            plt.title('Average Sample')
                            plt.axis('off')

                            # Right subplot: overlay image
                            plt.subplot(1, 2, 2)
                            plt.imshow(overlay)
                            plt.title('Average Heatmap Overlay')
                            plt.axis('off')
                            plt.tight_layout()

                            plt.savefig(overlay_path, dpi=300)
                            logger.info(f"Saved average overlay for {class_names[class_idx]} at {overlay_path}")

                            plt.close()
                        except Exception as e:
                            logger.warning(f"Error saving average overlay for class {class_idx}: {e}")
                            plt.close()
                except Exception as e:
                    logger.warning(f"Error processing class {class_idx}: {e}")
                    plt.close()
        except Exception as e:
            logger.error(f"Error in create_average_overlays: {e}")
            plt.close('all')  # Ensure all figures are closed

    def generate_family_heatmaps(self, X_data, y_data, family_labels, layer_name=None, num_classes=2):
        """
        Generate Grad-CAM heatmaps for each family within each class.

        Args:
            X_data: Input images as a NumPy array.
            y_data: Labels for the input images (binary: malware/benign).
            family_labels: Array of family labels for each sample (only relevant for malware).
            layer_name: Name of the layer to use for Grad-CAM (if None, use the last conv layer).
            num_classes: Number of classes (default is 2 for binary classification).

        Returns:
            Dictionary containing average heatmaps for each class-family combination.
        """
        try:
            if not self.conv_layers:
                logger.warning("No convolutional layers found in the model")
                return {}

            if layer_name is None:
                # Use the last convolutional layer by default
                layer_name = self.conv_layers[-1]

            logger.info(f"Generating family-based Grad-CAM heatmaps for layer: {layer_name}")

            # Predict class probabilities using the model
            try:
                y_pred_prob = self.model.predict(X_data)

                # Handle different model output formats
                if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
                    # For multi-class, get the predicted class with highest probability
                    y_pred = np.argmax(y_pred_prob, axis=1)
                else:
                    # For binary classification with sigmoid output
                    y_pred_prob = y_pred_prob.flatten()
                    y_pred = (y_pred_prob > 0.5).astype(int)
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return {}

            # Get unique family labels
            unique_families = np.unique(family_labels)
            logger.info(f"Found {len(unique_families)} unique malware families")

            # Initialize dictionaries to store heatmaps for each class-family combination
            family_heatmaps = {}
            correct_family_heatmaps = {}

            # Set a limit on the number of samples to process
            max_samples = min(len(X_data), 200)  # Increased limit for family analysis

            # Compute Grad-CAM for each sample and group by class and family
            for i in range(max_samples):
                try:
                    # Ensure the class index is within the valid range
                    class_idx = int(y_data[i]) % num_classes
                    family = family_labels[i]

                    # Skip if family is None or empty (likely for benign samples)
                    if family is None or family == "":
                        # For benign samples, use a special key
                        if class_idx == 0:  # Assuming 0 is benign
                            key = "benign"
                        else:
                            continue
                    else:
                        # For malware samples, use family name
                        key = family

                    # Compute GradCAM for this sample
                    cam = self.compute_gradcam(X_data[i], layer_name)

                    # Skip if CAM computation failed
                    if cam.size == 0 or np.all(cam == 0):
                        continue

                    # Store heatmap in appropriate dictionary based on family
                    if key not in family_heatmaps:
                        family_heatmaps[key] = []
                    family_heatmaps[key].append(cam)

                    # Also track correctly classified samples
                    if y_pred[i] == y_data[i]:
                        if key not in correct_family_heatmaps:
                            correct_family_heatmaps[key] = []
                        correct_family_heatmaps[key].append(cam)

                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue

            # Compute average heatmaps for each class-family combination
            avg_heatmaps = {}

            # Process all samples regardless of classification result
            for key, heatmaps in family_heatmaps.items():
                if heatmaps:
                    avg_heatmaps[f'family_{key}'] = np.mean(heatmaps, axis=0)

            # Process correctly classified samples
            for key, heatmaps in correct_family_heatmaps.items():
                if heatmaps:
                    avg_heatmaps[f'correct_family_{key}'] = np.mean(heatmaps, axis=0)

            # Save average heatmaps if an output directory is provided
            if self.output_dir:
                family_dir = os.path.join(self.output_dir, 'gradcam', 'family_analysis')
                os.makedirs(family_dir, exist_ok=True)

                for name, heatmap in avg_heatmaps.items():
                    try:
                        np.save(
                            os.path.join(family_dir, f'{name}_{layer_name}_heatmap.npy'),
                            heatmap
                        )
                    except Exception as e:
                        logger.warning(f"Error saving family heatmap {name}: {e}")

            return avg_heatmaps

        except Exception as e:
            logger.error(f"Error generating family heatmaps: {e}")
            logger.error(traceback.format_exc())
            return {}

    def visualize_family_heatmaps(self, family_heatmaps, layer_name=None):
        """
        Visualize family-based Grad-CAM heatmaps.

        Args:
            family_heatmaps: Dictionary of heatmaps from generate_family_heatmaps.
            layer_name: Name of the layer used for Grad-CAM.
        """
        try:
            if not family_heatmaps:
                logger.warning("No family heatmaps to visualize")
                return

            if layer_name is None:
                layer_name = 'conv'

            # Initialize family_viz_dir with a default value of None
            family_viz_dir = None

            # Create a directory for family visualizations
            if self.output_dir:
                try:
                    family_viz_dir = os.path.join(self.output_dir, 'gradcam', 'family_visualizations')
                    os.makedirs(family_viz_dir, exist_ok=True)
                    logger.info(f"Created family visualization directory at {family_viz_dir}")
                except Exception as e:
                    logger.warning(f"Error creating family visualization directory: {e}")
                    # Keep family_viz_dir as None if directory creation fails

            # Process each family heatmap
            for name, heatmap in family_heatmaps.items():
                try:
                    # Resize heatmap to the standard image size
                    heatmap_resized = cv2.resize(heatmap, (self.img_size, self.img_size))

                    plt.figure(figsize=(10, 8))
                    plt.imshow(heatmap_resized, cmap='jet')
                    plt.colorbar(label='Activation')

                    # Clean up the family name for title
                    if 'family_' in name:
                        title_name = name.replace('family_', '')
                    elif 'correct_family_' in name:
                        title_name = name.replace('correct_family_', 'Correctly Classified ')
                    else:
                        title_name = name

                    plt.title(f'Average Heatmap for {title_name}')
                    plt.axis('off')
                    plt.tight_layout()

                    # Only try to save if family_viz_dir is defined
                    if family_viz_dir is not None:
                        try:
                            # Sanitize filename
                            safe_name = name.replace('/', '_').replace('\\', '_')
                            save_path = os.path.join(family_viz_dir, f'{safe_name}_{layer_name}_heatmap.png')
                            plt.savefig(save_path, dpi=300)
                            logger.info(f"Saved family heatmap visualization to {save_path}")
                        except Exception as e:
                            logger.warning(f"Error saving family heatmap visualization: {e}")

                    plt.close()
                except Exception as e:
                    logger.warning(f"Error visualizing family heatmap {name}: {e}")
                    plt.close()
        except Exception as e:
            logger.error(f"Error in visualize_family_heatmaps: {e}")
            plt.close('all')

    def create_family_sample_overlays(self, X_samples, y_samples, family_labels, layer_name=None, num_samples=3):
        """
        Create overlays of Grad-CAM heatmaps on sample images for each family.

        Args:
            X_samples: Sample images as a NumPy array.
            y_samples: Labels for the sample images.
            family_labels: Family labels for the samples.
            layer_name: Name of the layer to use for Grad-CAM.
            num_samples: Number of samples per family to visualize.
        """
        try:
            if not self.conv_layers:
                logger.warning("No convolutional layers found in the model")
                return

            if layer_name is None:
                layer_name = self.conv_layers[-1]

            # Initialize family_overlay_dir with a default value of None
            family_overlay_dir = None

            if self.output_dir:
                try:
                    family_overlay_dir = os.path.join(self.output_dir, 'gradcam', 'family_overlays')
                    os.makedirs(family_overlay_dir, exist_ok=True)
                    logger.info(f"Created family overlay directory at {family_overlay_dir}")
                except Exception as e:
                    logger.warning(f"Error creating family overlay directory: {e}")
                    # Keep family_overlay_dir as None if directory creation fails

            # Group sample indices by family
            family_samples = {}
            for i in range(len(X_samples)):
                try:
                    family = family_labels[i]
                    class_idx = int(y_samples[i])

                    # Skip if family is None or empty (likely for benign samples)
                    if family is None or family == "":
                        # For benign samples, use a special key
                        if class_idx == 0:  # Assuming 0 is benign
                            key = "benign"
                        else:
                            continue
                    else:
                        # For malware samples, use family name
                        key = family

                    if key not in family_samples:
                        family_samples[key] = []
                    family_samples[key].append(i)
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue

            # Process each family (limit to num_samples per family)
            for family, sample_indices in family_samples.items():
                try:
                    # Limit samples to process
                    sample_indices = sample_indices[:min(num_samples, len(sample_indices))]

                    for i, idx in enumerate(sample_indices):
                        try:
                            img = X_samples[idx]
                            cam = self.compute_gradcam(img, layer_name)

                            # Skip if CAM computation failed
                            if cam.size == 0 or np.all(cam == 0):
                                continue

                            plt.figure(figsize=(12, 5))
                            try:
                                self._create_overlay_subplot(plt, 0, img, cam,
                                                             f"{family} - Sample {i + 1}")
                            except Exception as e:
                                logger.warning(f"Error creating overlay subplot: {e}")
                                plt.close()
                                continue

                            plt.tight_layout()

                            # Only try to save if family_overlay_dir is defined
                            if family_overlay_dir is not None:
                                try:
                                    # Sanitize filename
                                    safe_family = family.replace('/', '_').replace('\\', '_')
                                    save_path = os.path.join(
                                        family_overlay_dir,
                                        f'{safe_family}_sample_{i + 1}_{layer_name}.png'
                                    )
                                    plt.savefig(save_path, dpi=300)
                                    logger.info(f"Saved family sample overlay to {save_path}")
                                except Exception as e:
                                    logger.warning(f"Error saving family sample overlay: {e}")

                            plt.close()
                        except Exception as e:
                            logger.warning(f"Error processing sample {idx}: {e}")
                            plt.close()
                except Exception as e:
                    logger.warning(f"Error processing family {family}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in create_family_sample_overlays: {e}")
            plt.close('all')  # Ensure all figures are closed


    def create_family_average_overlays(self, X_data, y_data, family_labels, avg_heatmaps, layer_name=None):
        """
        Create overlays using the average image and the averaged heatmap for each family,
        computed over all samples in the family.

        Args:
            X_data: Input images as a NumPy array.
            y_data: Corresponding labels.
            family_labels: Family labels for the samples.
            avg_heatmaps: Dictionary of average heatmaps from generate_family_heatmaps.
            layer_name: Name of the layer used for Grad-CAM.
        """
        try:
            if not avg_heatmaps:
                logger.warning("No average family heatmaps provided")
                return

            # Initialize avg_overlay_dir with a default value of None
            avg_overlay_dir = None

            # Create directory for average family overlays
            if self.output_dir:
                try:
                    avg_overlay_dir = os.path.join(self.output_dir, 'gradcam', 'family_avg_overlays')
                    os.makedirs(avg_overlay_dir, exist_ok=True)
                    logger.info(f"Created family average overlays directory at {avg_overlay_dir}")
                except Exception as e:
                    logger.warning(f"Error creating family average overlays directory: {e}")
                    # Keep avg_overlay_dir as None if directory creation fails

            # Group images by family
            family_images = {}
            for i in range(len(X_data)):
                try:
                    family = family_labels[i]
                    class_idx = int(y_data[i])

                    # Skip if family is None or empty (likely for benign samples)
                    if family is None or family == "":
                        # For benign samples, use a special key
                        if class_idx == 0:  # Assuming 0 is benign
                            key = "benign"
                        else:
                            continue
                    else:
                        # For malware samples, use family name
                        key = family

                    if key not in family_images:
                        family_images[key] = []
                    family_images[key].append(X_data[i])
                except Exception as e:
                    logger.warning(f"Error grouping sample {i}: {e}")
                    continue

            # Process each family
            for family, images in family_images.items():
                try:
                    if not images:
                        logger.warning(f"No images found for family {family}")
                        continue

                    # Compute average image for this family
                    avg_image = np.mean(images, axis=0)

                    # Retrieve the corresponding average heatmap
                    heatmap_key = f'family_{family}'
                    if heatmap_key not in avg_heatmaps:
                        logger.warning(f"No average heatmap found for family {family}")
                        continue

                    heatmap = avg_heatmaps[heatmap_key]

                    # Resize heatmap to match the average image dimensions
                    if len(avg_image.shape) == 3:
                        height, width = avg_image.shape[:2]
                    else:
                        height, width = avg_image.shape

                    heatmap_resized = cv2.resize(heatmap, (width, height))

                    # Convert heatmap to color
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                    # Convert average image to uint8 (assuming it is in [0,1] range)
                    avg_image_uint8 = np.uint8(avg_image * 255)
                    if len(avg_image.shape) == 2:
                        avg_image_uint8 = cv2.cvtColor(avg_image_uint8, cv2.COLOR_GRAY2BGR)
                    elif len(avg_image.shape) == 3 and avg_image.shape[-1] == 1:
                        avg_image_uint8 = cv2.cvtColor(avg_image_uint8, cv2.COLOR_GRAY2BGR)

                    # Create overlay from the average image and averaged heatmap
                    overlay = cv2.addWeighted(colored_heatmap, 0.4, avg_image_uint8, 0.6, 0)
                    overlay = np.clip(overlay, 0, 255).astype('uint8')

                    # Create side-by-side visualization
                    # Only try to save if avg_overlay_dir is defined
                    if avg_overlay_dir is not None:
                        try:
                            # Sanitize filename
                            safe_family = family.replace('/', '_').replace('\\', '_')
                            overlay_path = os.path.join(
                                avg_overlay_dir,
                                f'avg_overlay_family_{safe_family}.png'
                            )

                            plt.figure(figsize=(12, 6))
                            # Left subplot: average sample image
                            plt.subplot(1, 2, 1)
                            # Display as grayscale if the image is single channel
                            if len(avg_image.shape) == 2 or (len(avg_image.shape) == 3 and avg_image.shape[-1] == 1):
                                plt.imshow(avg_image.squeeze(), cmap='gray')
                            else:
                                plt.imshow(avg_image)
                            plt.title(f'Average Sample - {family}')
                            plt.axis('off')

                            # Right subplot: overlay image
                            plt.subplot(1, 2, 2)
                            plt.imshow(overlay)
                            plt.title(f'Average Heatmap Overlay - {family}')
                            plt.axis('off')
                            plt.tight_layout()

                            plt.savefig(overlay_path, dpi=300)
                            logger.info(f"Saved average overlay for family {family} at {overlay_path}")

                            plt.close()
                        except Exception as e:
                            logger.warning(f"Error saving average overlay for family {family}: {e}")
                            plt.close()
                    else:
                        # We still want to generate and show the visualization even if we can't save it
                        plt.figure(figsize=(12, 6))
                        # Left subplot: average sample image
                        plt.subplot(1, 2, 1)
                        # Display as grayscale if the image is single channel
                        if len(avg_image.shape) == 2 or (len(avg_image.shape) == 3 and avg_image.shape[-1] == 1):
                            plt.imshow(avg_image.squeeze(), cmap='gray')
                        else:
                            plt.imshow(avg_image)
                        plt.title(f'Average Sample - {family}')
                        plt.axis('off')

                        # Right subplot: overlay image
                        plt.subplot(1, 2, 2)
                        plt.imshow(overlay)
                        plt.title(f'Average Heatmap Overlay - {family}')
                        plt.axis('off')
                        plt.tight_layout()

                        # Just display, don't save
                        logger.info(f"Generated average overlay for family {family} (not saved)")
                        plt.close()
                except Exception as e:
                    logger.warning(f"Error processing family {family}: {e}")
                    plt.close()
        except Exception as e:
            logger.error(f"Error in create_family_average_overlays: {e}")
            plt.close('all')  # Ensure all figures are closed


def analyze_by_family(args, model=None, X_test=None, y_test=None, data_handler=None):
    """Generate Grad-CAM visualizations grouped by malware family"""
    print("=== Starting Family-Based Grad-CAM Analysis ===")

    # Create output directory
    output_dir = args.results_dir if hasattr(args, 'results_dir') and args.results_dir else 'results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using output directory: {output_dir}")

    # Load model if not provided
    if model is None:
        custom_objects = {'CastLayer': CastLayer}

        # Check if model path is provided
        if args.model_path is None:
            # Try to find a model in the results directory
            if os.path.exists(os.path.join(output_dir, 'model')):
                model_dir = os.path.join(output_dir, 'model')
                # Check for final model first, then best model
                if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
                    args.model_path = os.path.join(model_dir, 'final_model.h5')
                elif os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                    args.model_path = os.path.join(model_dir, 'best_model.h5')

        if args.model_path is None:
            raise ValueError("Model path not provided and no model found in results directory")

        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    # Load test data if not provided
    if X_test is None or y_test is None or data_handler is None:
        data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Try to load from splits first
        if hasattr(args, 'use_splits') and args.use_splits and os.path.exists(
                os.path.join(output_dir, 'data_splits.pkl')):
            success = data_handler.load_from_splits(output_dir)
            if not success:
                data_handler.load_and_preprocess_data()
        else:
            data_handler.load_and_preprocess_data()

        # Load test data
        X_test, y_test = data_handler.load_test_data()

    # Check if family labels are available
    if not hasattr(data_handler, 'family_labels') or data_handler.family_labels is None:
        print("Error: Family labels not found in data_handler. Cannot perform family-based analysis.")
        return

    # Get family labels for test data
    test_family_labels = []
    for file_path in data_handler.test_files:
        # Use the file_to_family_map to get the family label
        family = data_handler.file_to_family_map.get(file_path, "unknown")
        test_family_labels.append(family)

    # Initialize Grad-CAM generator
    gradcam_generator = GradCAMGenerator(
        model=model,
        img_size=args.img_size,
        output_dir=output_dir
    )

    # Get layer names
    conv_layers = [layer.name for layer in model.layers
                   if isinstance(layer, tf.keras.layers.Conv2D)]

    # If no specific layer is provided, use the last few conv layers
    target_layers = [args.layer_name] if hasattr(args, 'layer_name') and args.layer_name else conv_layers[-3:]

    # Initialize results dictionaries to avoid reference before assignment warnings
    class_heatmaps = {}
    family_heatmaps = {}

    for layer_name in target_layers:
        print(f"Analyzing layer: {layer_name}")

        # Generate standard heatmaps by class
        print("Generating class-based heatmaps...")
        current_class_heatmaps = gradcam_generator.generate_heatmaps(
            X_test, y_test, layer_name, num_classes=2
        )

        # Store current layer's class heatmaps in results
        class_heatmaps[layer_name] = current_class_heatmaps

        # Visualize standard heatmaps
        gradcam_generator.visualize_heatmaps(
            current_class_heatmaps, layer_name, class_names=['Benign', 'Malware']
        )

        # Generate family-based heatmaps
        print("Generating family-based heatmaps...")
        current_family_heatmaps = gradcam_generator.generate_family_heatmaps(
            X_test, y_test, test_family_labels, layer_name, num_classes=2
        )

        # Store current layer's family heatmaps in results
        family_heatmaps[layer_name] = current_family_heatmaps

        # Visualize family heatmaps
        print("Visualizing family-based heatmaps...")
        gradcam_generator.visualize_family_heatmaps(
            current_family_heatmaps, layer_name
        )

        # Create sample overlays for each family
        print("Creating sample overlays for each family...")
        gradcam_generator.create_family_sample_overlays(
            X_test, y_test, test_family_labels, layer_name,
            num_samples=min(3, len(X_test))
        )

        # Create average overlays for each family
        print("Creating average overlays for each family...")
        gradcam_generator.create_family_average_overlays(
            X_test, y_test, test_family_labels, current_family_heatmaps, layer_name
        )

    print(f"Family-based Grad-CAM analysis completed. Results saved to {output_dir}/gradcam/")
    return {
        'class_heatmaps': class_heatmaps,
        'family_heatmaps': family_heatmaps
    }


class HeatmapVisualizer:
    def __init__(self, output_dir=None):
        """
        Initialize the heatmap visualizer.

        Args:
            output_dir: Directory to save visualizations.
        """
        self.output_dir = output_dir

        try:
            if output_dir:
                os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create output directory: {e}")

    def batch_visualize_heatmaps(self, heatmap_dir):
        """
        Visualize all .npy heatmap files in a directory.

        Args:
            heatmap_dir: Directory containing .npy heatmap files.

        Returns:
            List of paths to the saved visualization images.
        """
        output_paths = []

        try:
            if not os.path.exists(heatmap_dir):
                logger.warning(f"Heatmap directory {heatmap_dir} does not exist")
                return output_paths

            npy_files = [f for f in os.listdir(heatmap_dir) if f.endswith('.npy')]

            if not npy_files:
                logger.info(f"No .npy files found in {heatmap_dir}")
                return output_paths

            for npy_file in npy_files:
                try:
                    heatmap_path = os.path.join(heatmap_dir, npy_file)
                    heatmap = np.load(heatmap_path)

                    plt.figure(figsize=(10, 8))
                    plt.imshow(heatmap, cmap='jet')

                    # Create colorbar with proper error handling
                    try:
                        sm = plt.cm.ScalarMappable(cmap='jet')
                        sm.set_array([])
                        plt.colorbar(sm, label='Activation')
                    except Exception as e:
                        logger.warning(f"Error adding colorbar to {npy_file}: {e}")

                    output_name = npy_file.replace('.npy', '.png')

                    if self.output_dir:
                        output_path = os.path.join(self.output_dir, 'visualizations', output_name)
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Saved heatmap visualization to {output_path}")
                        output_paths.append(output_path)

                    plt.close()
                except Exception as e:
                    logger.warning(f"Error processing heatmap file {npy_file}: {e}")
                    plt.close()

            return output_paths
        except Exception as e:
            logger.error(f"Error in batch_visualize_heatmaps: {e}")
            plt.close('all')
            return output_paths

    def batch_create_overlays(self, sample_images_dir, heatmap_dir, class_dirs=None):
        """
        Create overlays for sample images using pre-saved heatmaps.

        Args:
            sample_images_dir: Directory containing sample images.
            heatmap_dir: Directory containing .npy heatmap files.
            class_dirs: Dictionary mapping class names to subdirectories.

        Returns:
            List of paths to the saved overlay images.
        """
        try:
            # This is a stub that could be implemented with proper error handling
            logger.info("Overlay creation not yet implemented")
            return []
        except Exception as e:
            logger.error(f"Error in batch_create_overlays: {e}")
            return []

