#!/usr/bin/env python3
"""
Feature Importance Analysis Module for Malware Classification System
Provides tools for visualizing feature importance, embeddings, and activations
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns


class FeatureAnalyzer:
    def __init__(self, model, data_handler, output_dir=None):
        """
        Initialize the feature analyzer.

        Args:
            model: Trained TensorFlow model
            data_handler: DataHandler instance containing the data
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.data_handler = data_handler
        self.output_dir = output_dir

        if output_dir:
            os.makedirs(os.path.join(output_dir, 'feature_analysis'), exist_ok=True)

    def extract_embeddings(self, X_data, layer_name=None):
        """
        Extract embeddings from a specific layer of the model.

        Args:
            X_data: Input data to extract embeddings from
            layer_name: Name of the layer to extract embeddings from (default: last layer before prediction)

        Returns:
            Numpy array of embeddings
        """
        # If no layer specified, use the layer before the final dense
        if layer_name is None:
            # Find the second to last layer (assuming the last layer is the output layer)
            for layer in reversed(self.model.layers[:-1]):
                if len(layer.output_shape) <= 2:  # Looking for a flat layer (dense or global pooling)
                    layer_name = layer.name
                    break

        if layer_name is None:
            raise ValueError("Could not find a suitable layer for embedding extraction")

        print(f"Extracting embeddings from layer: {layer_name}")

        # Create a new model that outputs the embeddings
        embedding_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer(layer_name).output
        )

        # Extract embeddings - process in batches to handle large datasets
        batch_size = 64
        n_samples = len(X_data)
        n_batches = (n_samples + batch_size - 1) // batch_size
        embeddings = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_embeddings = embedding_model.predict(X_data[start_idx:end_idx])
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def visualize_tsne(self, X_data, y_data, layer_name=None, perplexity=30, n_iter=1000):
        """
        Create t-SNE visualization of the embeddings.

        Args:
            X_data: Input data
            y_data: Labels for input data
            layer_name: Layer to extract embeddings from
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations for t-SNE

        Returns:
            Path to the saved visualization
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(X_data, layer_name)

        # Flatten embeddings if they're not already flat
        if len(embeddings.shape) > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        print(f"Running t-SNE on {embeddings.shape[0]} samples with {embeddings.shape[1]} features...")

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create visualization
        plt.figure(figsize=(12, 10))
        classes = np.unique(y_data)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

        for i, cls in enumerate(classes):
            plt.scatter(
                embeddings_2d[y_data == cls, 0],
                embeddings_2d[y_data == cls, 1],
                c=[colors[i]],
                label=f'Class {cls}',
                alpha=0.7
            )

        plt.legend()
        plt.title(f't-SNE Visualization of Embeddings from {layer_name or "last layer"}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()

        # Save visualization
        if self.output_dir:
            output_path = os.path.join(
                self.output_dir, 'feature_analysis',
                f'tsne_{layer_name or "last_layer"}.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE visualization saved to {output_path}")
            return output_path
        else:
            plt.show()
            return None

    def visualize_umap(self, X_data, y_data, layer_name=None, n_neighbors=15, min_dist=0.1):
        """
        Create UMAP visualization of the embeddings.

        Args:
            X_data: Input data
            y_data: Labels for input data
            layer_name: Layer to extract embeddings from
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP minimum distance parameter

        Returns:
            Path to the saved visualization
        """
        # Extract embeddings
        embeddings = self.extract_embeddings(X_data, layer_name)

        # Flatten embeddings if they're not already flat
        if len(embeddings.shape) > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        print(f"Running UMAP on {embeddings.shape[0]} samples with {embeddings.shape[1]} features...")

        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

        # Create visualization
        plt.figure(figsize=(12, 10))
        classes = np.unique(y_data)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

        for i, cls in enumerate(classes):
            plt.scatter(
                embeddings_2d[y_data == cls, 0],
                embeddings_2d[y_data == cls, 1],
                c=[colors[i]],
                label=f'Class {cls}',
                alpha=0.7
            )

        plt.legend()
        plt.title(f'UMAP Visualization of Embeddings from {layer_name or "last layer"}')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.tight_layout()

        # Save visualization
        if self.output_dir:
            output_path = os.path.join(
                self.output_dir, 'feature_analysis',
                f'umap_{layer_name or "last_layer"}.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"UMAP visualization saved to {output_path}")
            return output_path
        else:
            plt.show()
            return None

    def visualize_layer_activations(self, X_sample, layer_names=None, sample_index=0):
        """
        Visualize activations of a specific layer for a single input sample.

        Args:
            X_sample: Single input sample or batch of samples
            layer_names: List of layer names to visualize (if None, use all conv layers)
            sample_index: Index to use if X_sample is a batch

        Returns:
            Dictionary mapping layer names to saved visualization paths
        """
        # If no layers specified, use all convolutional layers
        if layer_names is None:
            layer_names = [layer.name for layer in self.model.layers
                           if isinstance(layer, tf.keras.layers.Conv2D)]

        # If X_sample is a batch, extract a single sample
        if len(X_sample.shape) == 4 and X_sample.shape[0] > 1:
            X_sample = X_sample[sample_index:sample_index + 1]

        output_paths = {}
        for layer_name in layer_names:
            # Create a model that outputs the layer activations
            activation_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=self.model.get_layer(layer_name).output
            )

            # Get activations
            activations = activation_model.predict(X_sample)

            # For a single sample, remove the batch dimension
            if activations.shape[0] == 1:
                activations = activations[0]

            # Create visualization
            n_filters = activations.shape[-1]
            n_cols = min(8, n_filters)
            n_rows = (n_filters + n_cols - 1) // n_cols

            plt.figure(figsize=(n_cols * 2, n_rows * 2))
            for i in range(n_filters):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(activations[:, :, i], cmap='viridis')
                plt.title(f'Filter {i}')
                plt.axis('off')

            plt.suptitle(f'Layer {layer_name} Activations')
            plt.tight_layout()

            # Save visualization
            if self.output_dir:
                output_path = os.path.join(
                    self.output_dir, 'feature_analysis',
                    f'activations_{layer_name}_sample_{sample_index}.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                output_paths[layer_name] = output_path

            plt.close()

        return output_paths

    def generate_class_activation_maps(self, X_data, y_data, n_samples=5):
        """
        Generate class activation maps for multiple samples of each class.

        Args:
            X_data: Input data
            y_data: Labels for input data
            n_samples: Number of samples per class to generate maps for

        Returns:
            Dictionary mapping class indices to lists of sample indices and activation maps
        """
        from visualization import GradCAMGenerator

        # Create a GradCAM generator
        gradcam_gen = GradCAMGenerator(
            model=self.model,
            output_dir=self.output_dir
        )

        # Group samples by class
        class_indices = {}
        for i, label in enumerate(y_data):
            class_idx = int(label)
            if class_idx not in class_indices:
                class_indices[class_idx] = []
            class_indices[class_idx].append(i)

        # For each class, generate activation maps for n_samples
        results = {}
        for class_idx, indices in class_indices.items():
            # Select a subset of samples
            selected_indices = indices[:n_samples]
            selected_samples = X_data[selected_indices]

            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break

            if last_conv_layer is None:
                print("No convolutional layer found in the model.")
                continue

            # Generate CAMs for each sample
            sample_cams = []
            for i, idx in enumerate(selected_indices):
                cam = gradcam_gen.compute_gradcam(X_data[idx], last_conv_layer)
                sample_cams.append(cam)

                # Create a visualization
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                if len(X_data[idx].shape) == 3 and X_data[idx].shape[-1] == 1:
                    plt.imshow(X_data[idx][:, :, 0], cmap='gray')
                else:
                    plt.imshow(X_data[idx], cmap='gray')
                plt.title(f'Class {class_idx} - Sample {i + 1}')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(cam, cmap='jet')
                plt.title('Class Activation Map')
                plt.axis('off')

                plt.tight_layout()

                if self.output_dir:
                    output_path = os.path.join(
                        self.output_dir, 'feature_analysis',
                        f'cam_class_{class_idx}_sample_{i + 1}.png'
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')

                plt.close()

            results[class_idx] = {
                'indices': selected_indices,
                'cams': sample_cams
            }

        return results

    def visualize_activation_maximization(self, layer_names=None, filter_indices=None, iterations=30):
        """
        Generate images that maximize activations of specific filters in the network.

        Args:
            layer_names: List of layer names to visualize (if None, use last conv layer)
            filter_indices: List of filter indices to visualize (if None, use first few filters)
            iterations: Number of gradient ascent iterations

        Returns:
            Dictionary mapping layer names and filter indices to visualization paths
        """
        # If no layers specified, use the last convolutional layer
        if layer_names is None:
            layer_names = []
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_names.append(layer.name)
                    break

        output_paths = {}

        # Process each layer
        for layer_name in layer_names:
            layer = self.model.get_layer(layer_name)

            # Determine filter indices to visualize
            if filter_indices is None:
                # Use first few filters
                n_filters = min(9, layer.filters)
                filter_indices = list(range(n_filters))

            # Set up the figure
            n_filters = len(filter_indices)
            n_cols = min(3, n_filters)
            n_rows = (n_filters + n_cols - 1) // n_cols
            plt.figure(figsize=(n_cols * 4, n_rows * 4))

            # Process each filter
            for i, filter_idx in enumerate(filter_indices):
                # Generate input that maximizes the filter activation
                img = self._generate_filter_visualization(layer_name, filter_idx, iterations)

                # Add to the plot
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(img[..., 0], cmap='viridis')
                plt.title(f'Filter {filter_idx}')
                plt.axis('off')

            plt.suptitle(f'Layer {layer_name} Filter Visualizations')
            plt.tight_layout()

            # Save visualization
            if self.output_dir:
                output_path = os.path.join(
                    self.output_dir, 'feature_analysis',
                    f'filter_viz_{layer_name}.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                output_paths[layer_name] = output_path

            plt.close()

        return output_paths

    def _generate_filter_visualization(self, layer_name, filter_index, iterations=30):
        """
        Generate an image that maximizes activations of a specific filter.

        Args:
            layer_name: Name of the layer containing the filter
            filter_index: Index of the filter to visualize
            iterations: Number of gradient ascent iterations

        Returns:
            Numpy array representing the generated image
        """
        # Create a model that outputs the target layer's activations
        layer = self.model.get_layer(layer_name)
        feature_extractor = tf.keras.models.Model(inputs=self.model.inputs,
                                                  outputs=layer.output)

        # Start from a random noise image
        input_shape = self.model.input_shape[1:]  # Exclude batch dimension
        img = np.random.random(input_shape) * 0.1 + 0.5  # Random noise centered at 0.5

        # Convert to tensor
        img = tf.Variable(tf.cast(img[np.newaxis], tf.float32))

        # Perform gradient ascent
        learning_rate = 1.0
        for _ in range(iterations):
            with tf.GradientTape() as tape:
                outputs = feature_extractor(img)
                # Loss is the mean activation of the target filter
                loss = tf.reduce_mean(outputs[..., filter_index])

            # Compute gradients and update the image
            grads = tape.gradient(loss, img)
            # Normalize gradients
            grads = tf.math.l2_normalize(grads)
            img.assign_add(grads * learning_rate)

            # Keep pixel values between 0 and 1
            img.assign(tf.clip_by_value(img, 0, 1))

        # Return the image as a numpy array
        return img.numpy()[0]