#!/usr/bin/env python3
"""
Robust Error Analysis Module for Malware Classification System
Provides tools for analyzing model errors, confusion matrices, and difficult examples
with comprehensive error handling to prevent training pipeline disruption
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ErrorAnalyzer")


class ErrorAnalyzer:
    def __init__(self, model, data_handler=None, output_dir=None):
        """
        Initialize the error analyzer.

        Args:
            model: Trained TensorFlow model
            data_handler: DataHandler instance containing the data
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.data_handler = data_handler
        self.output_dir = output_dir

        if output_dir:
            try:
                os.makedirs(os.path.join(output_dir, 'error_analysis'), exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create output directory: {e}")

    def analyze_errors(self, X_data, y_true, class_names=None, threshold=0.5):
        """
        Perform comprehensive error analysis on the provided data.

        Args:
            X_data: Input data to analyze
            y_true: Ground truth labels
            class_names: List of class names (default: ['Benign', 'Malware'])
            threshold: Classification threshold for binary classification

        Returns:
            Dictionary containing error analysis results
        """
        try:
            if class_names is None:
                class_names = ['Benign', 'Malware']

            # Make predictions
            try:
                y_pred_prob = self.model.predict(X_data)

                # Handle different prediction shapes
                if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
                    # For multi-class models, get the probability of the positive class
                    logger.info("Multi-dimensional output detected, using positive class probabilities")
                    y_pred_prob = y_pred_prob[:, 1]

                y_pred_prob = y_pred_prob.flatten()
                y_pred = (y_pred_prob > threshold).astype(int).flatten()
                y_true = y_true.flatten()
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return {'error': f"Prediction failed: {str(e)}"}

            # Get indices of correct and incorrect predictions
            correct_indices = np.where(y_pred == y_true)[0]
            error_indices = np.where(y_pred != y_true)[0]

            # Calculate confidence scores
            confidence = np.abs(y_pred_prob - 0.5) * 2  # Scale to [0, 1]

            # Find the most confident correct and incorrect predictions
            if len(correct_indices) > 0:
                most_confident_correct = correct_indices[np.argsort(-confidence[correct_indices].flatten())[:10]]
            else:
                most_confident_correct = []

            if len(error_indices) > 0:
                most_confident_errors = error_indices[np.argsort(-confidence[error_indices].flatten())[:10]]
            else:
                most_confident_errors = []

            # Find the least confident correct and incorrect predictions
            if len(correct_indices) > 0:
                least_confident_correct = correct_indices[np.argsort(confidence[correct_indices].flatten())[:10]]
            else:
                least_confident_correct = []

            if len(error_indices) > 0:
                least_confident_errors = error_indices[np.argsort(confidence[error_indices].flatten())[:10]]
            else:
                least_confident_errors = []

            # Find predictions near the decision boundary
            boundary_indices = np.argsort(np.abs(y_pred_prob - threshold).flatten())[:20]

            # Generate confusion matrix
            try:
                cm = confusion_matrix(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not compute confusion matrix: {e}")
                cm = np.array([[0, 0], [0, 0]])

            # Create classification report
            try:
                report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            except Exception as e:
                logger.warning(f"Could not create classification report: {e}")
                report = {}

            # Save results
            results = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'correct_indices': correct_indices,
                'error_indices': error_indices,
                'most_confident_correct': most_confident_correct,
                'most_confident_errors': most_confident_errors,
                'least_confident_correct': least_confident_correct,
                'least_confident_errors': least_confident_errors,
                'boundary_indices': boundary_indices,
                'confusion_matrix': cm,
                'classification_report': report,
                'confidence': confidence,
            }

            # Generate visualizations (wrapped in try/except to prevent failures)
            try:
                self._plot_confusion_matrix(cm, class_names)
            except Exception as e:
                logger.warning(f"Error plotting confusion matrix: {e}")

            try:
                self._plot_error_distribution(y_true, y_pred, y_pred_prob, class_names)
            except Exception as e:
                logger.warning(f"Error plotting error distribution: {e}")

            try:
                self._visualize_examples(X_data, y_true, y_pred, y_pred_prob, results)
            except Exception as e:
                logger.warning(f"Error visualizing examples: {e}")

            try:
                self._analyze_decision_boundary(X_data, y_true, y_pred, y_pred_prob, boundary_indices)
            except Exception as e:
                logger.warning(f"Error analyzing decision boundary: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in analyze_errors: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}

    def _plot_confusion_matrix(self, cm, class_names):
        """
        Plot and save confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: List of class names
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'error_analysis', 'confusion_matrix.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {output_path}")

            plt.close()

            # Normalized confusion matrix
            plt.figure(figsize=(10, 8))
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0 and Inf with large finite numbers

            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Normalized Confusion Matrix')

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'error_analysis', 'confusion_matrix_normalized.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Normalized confusion matrix saved to {output_path}")

            plt.close()
        except Exception as e:
            logger.warning(f"Error in _plot_confusion_matrix: {e}")
            plt.close('all')  # Ensure all figures are closed

    def _plot_error_distribution(self, y_true, y_pred, y_pred_prob, class_names):
        """
        Plot distribution of prediction probabilities for correct and incorrect predictions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_prob: Prediction probabilities
            class_names: List of class names
        """
        try:
            # Separate correct and incorrect predictions
            correct_mask = y_true == y_pred
            incorrect_mask = ~correct_mask

            # Check if we have any correct/incorrect predictions
            if not np.any(correct_mask) and not np.any(incorrect_mask):
                logger.warning("No prediction data to plot distributions")
                return

            # Plot histograms of prediction probabilities
            plt.figure(figsize=(12, 8))

            if np.any(correct_mask):
                plt.hist(y_pred_prob[correct_mask], alpha=0.5, bins=20,
                         label='Correct Predictions', color='green')

            if np.any(incorrect_mask):
                plt.hist(y_pred_prob[incorrect_mask], alpha=0.5, bins=20,
                         label='Incorrect Predictions', color='red')

            plt.xlabel('Prediction Probability')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Probabilities')
            plt.legend()
            plt.grid(alpha=0.3)

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'error_analysis', 'probability_distribution.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Probability distribution saved to {output_path}")

            plt.close()

            # Plot prediction confidence distribution
            confidence = np.abs(y_pred_prob - 0.5) * 2  # Scale to [0, 1]

            plt.figure(figsize=(12, 8))

            if np.any(correct_mask):
                plt.hist(confidence[correct_mask], alpha=0.5, bins=20,
                         label='Correct Predictions', color='green')

            if np.any(incorrect_mask):
                plt.hist(confidence[incorrect_mask], alpha=0.5, bins=20,
                         label='Incorrect Predictions', color='red')

            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Confidence')
            plt.legend()
            plt.grid(alpha=0.3)

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'error_analysis', 'confidence_distribution.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confidence distribution saved to {output_path}")

            plt.close()

            # Create error rate by confidence bin
            try:
                bins = np.linspace(0, 1, 11)
                bin_indices = np.digitize(confidence, bins) - 1
                bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

                bin_correct = np.zeros(len(bins) - 1)
                bin_total = np.zeros(len(bins) - 1)

                for i in range(len(y_true)):
                    bin_idx = bin_indices[i]
                    bin_total[bin_idx] += 1
                    if correct_mask[i]:
                        bin_correct[bin_idx] += 1

                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    bin_error_rate = 1 - (bin_correct / np.maximum(bin_total, 1))
                    bin_error_rate = np.nan_to_num(bin_error_rate)

                bin_centers = (bins[:-1] + bins[1:]) / 2

                plt.figure(figsize=(12, 8))
                plt.bar(bin_centers, bin_error_rate, width=0.08, alpha=0.7)

                for i, (center, error, total) in enumerate(zip(bin_centers, bin_error_rate, bin_total)):
                    if total > 0:
                        plt.text(center, error + 0.02, f'{int(total)}', ha='center')

                plt.xlabel('Confidence')
                plt.ylabel('Error Rate')
                plt.title('Error Rate by Confidence Level')
                plt.grid(alpha=0.3)
                plt.ylim(0, 1)

                if self.output_dir:
                    output_path = os.path.join(self.output_dir, 'error_analysis', 'error_rate_by_confidence.png')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Error rate by confidence saved to {output_path}")

                plt.close()
            except Exception as e:
                logger.warning(f"Error creating error rate by confidence bins: {e}")
                plt.close()

        except Exception as e:
            logger.warning(f"Error in _plot_error_distribution: {e}")
            plt.close('all')  # Ensure all figures are closed

    def _visualize_examples(self, X_data, y_true, y_pred, y_pred_prob, results):
        """
        Visualize examples of correct and incorrect predictions.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_prob: Prediction probabilities
            results: Dictionary of analysis results
        """
        try:
            # Create directories for example images
            if self.output_dir:
                try:
                    os.makedirs(os.path.join(self.output_dir, 'error_analysis', 'difficult_examples'), exist_ok=True)
                    os.makedirs(os.path.join(self.output_dir, 'error_analysis', 'confident_examples'), exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not create directories for example images: {e}")
                    return

            # Visualize the most confident errors
            try:
                self._create_example_grid(
                    X_data, y_true, y_pred, y_pred_prob,
                    results['most_confident_errors'],
                    'Most Confident Errors',
                    os.path.join(self.output_dir, 'error_analysis', 'difficult_examples', 'most_confident_errors.png')
                )
            except Exception as e:
                logger.warning(f"Error visualizing most confident errors: {e}")

            # Visualize the least confident correct predictions
            try:
                self._create_example_grid(
                    X_data, y_true, y_pred, y_pred_prob,
                    results['least_confident_correct'],
                    'Least Confident Correct Predictions',
                    os.path.join(self.output_dir, 'error_analysis', 'difficult_examples', 'least_confident_correct.png')
                )
            except Exception as e:
                logger.warning(f"Error visualizing least confident correct predictions: {e}")

            # Visualize examples near the decision boundary
            try:
                self._create_example_grid(
                    X_data, y_true, y_pred, y_pred_prob,
                    results['boundary_indices'],
                    'Examples Near Decision Boundary',
                    os.path.join(self.output_dir, 'error_analysis', 'difficult_examples', 'boundary_examples.png')
                )
            except Exception as e:
                logger.warning(f"Error visualizing boundary examples: {e}")

            # Visualize most confident correct predictions
            try:
                self._create_example_grid(
                    X_data, y_true, y_pred, y_pred_prob,
                    results['most_confident_correct'],
                    'Most Confident Correct Predictions',
                    os.path.join(self.output_dir, 'error_analysis', 'confident_examples', 'most_confident_correct.png')
                )
            except Exception as e:
                logger.warning(f"Error visualizing most confident correct predictions: {e}")

            # Create a difficult examples catalog
            try:
                self._create_error_catalog(X_data, y_true, y_pred, y_pred_prob, results)
            except Exception as e:
                logger.warning(f"Error creating error catalog: {e}")

        except Exception as e:
            logger.warning(f"Error in _visualize_examples: {e}")

    def _create_example_grid(self, X_data, y_true, y_pred, y_pred_prob, indices, title, output_path=None,
                             class_names=None):
        """
        Create a grid visualization of example images.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_prob: Prediction probabilities
            indices: Indices of examples to visualize
            title: Title for the grid
            output_path: Path to save the visualization
            class_names: List of class names (default: ['Benign', 'Malware'])
        """
        try:
            if class_names is None:
                class_names = ['Benign', 'Malware']

            if len(indices) == 0:
                logger.info(f"No examples found for {title}")
                return

            n_examples = min(len(indices), 9)
            indices = indices[:n_examples]

            # Calculate grid dimensions
            n_cols = min(3, n_examples)
            n_rows = (n_examples + n_cols - 1) // n_cols

            plt.figure(figsize=(n_cols * 4, n_rows * 4))

            for i, idx in enumerate(indices):
                plt.subplot(n_rows, n_cols, i + 1)

                try:
                    # Check if index is valid
                    if idx >= len(X_data):
                        logger.warning(f"Index {idx} out of range for X_data (length {len(X_data)})")
                        continue

                    # Display the image based on its shape
                    if len(X_data.shape) == 4:  # For multi-channel images
                        if X_data.shape[3] == 1:  # Single channel
                            plt.imshow(X_data[idx, :, :, 0], cmap='gray')
                        elif X_data.shape[3] == 3:  # RGB
                            plt.imshow(X_data[idx])
                        else:  # Handle other channel counts
                            logger.info(f"Unusual channel count: {X_data.shape[3]}, displaying first channel")
                            plt.imshow(X_data[idx, :, :, 0], cmap='gray')
                    elif len(X_data.shape) == 3:  # For single-channel or grayscale
                        plt.imshow(X_data[idx], cmap='gray')
                    elif len(X_data.shape) == 2:  # For 1D data (e.g., feature vectors)
                        plt.bar(range(len(X_data[idx])), X_data[idx])
                        plt.title('Feature Vector')
                    else:
                        logger.warning(f"Unexpected data shape: {X_data.shape}")
                        continue

                    # Create label with true and predicted classes
                    true_class = class_names[int(y_true[idx])] if int(y_true[idx]) < len(class_names) else str(
                        int(y_true[idx]))
                    pred_class = class_names[int(y_pred[idx])] if int(y_pred[idx]) < len(class_names) else str(
                        int(y_pred[idx]))
                    confidence = float(abs(y_pred_prob[idx] - 0.5) * 2)
                    prob = float(y_pred_prob[idx])

                    color = 'green' if y_true[idx] == y_pred[idx] else 'red'
                    plt.title(f"True: {true_class}\nPred: {pred_class}\nProb: {prob:.3f}\nConf: {confidence:.3f}",
                              color=color, fontsize=12)
                    plt.axis('off')
                except Exception as e:
                    logger.warning(f"Error displaying example at index {idx}: {e}")
                    continue

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            if output_path and self.output_dir:
                try:
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Example grid saved to {output_path}")
                except Exception as e:
                    logger.warning(f"Error saving example grid to {output_path}: {e}")

            plt.close()
        except Exception as e:
            logger.warning(f"Error in _create_example_grid: {e}")
            plt.close('all')  # Ensure all figures are closed

    def _create_error_catalog(self, X_data, y_true, y_pred, y_pred_prob, results):
        """
        Create a catalog of all misclassified examples.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_prob: Prediction probabilities
            results: Dictionary of analysis results
        """
        try:
            error_indices = results['error_indices']

            if len(error_indices) == 0:
                logger.info("No errors found for the error catalog")
                return

            # Create a directory for error catalog
            if self.output_dir:
                try:
                    catalog_dir = os.path.join(self.output_dir, 'error_analysis', 'error_catalog')
                    os.makedirs(catalog_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not create error catalog directory: {e}")
                    return

                # Limit the number of examples to save to prevent overwhelming the disk
                max_examples = min(len(error_indices), 100)
                error_indices = error_indices[:max_examples]

                # Save each misclassified example
                error_data = []
                for i, idx in enumerate(error_indices):
                    try:
                        true_label = int(y_true[idx])
                        pred_label = int(y_pred[idx])
                        prob = float(y_pred_prob[idx])
                        confidence = float(abs(prob - 0.5) * 2)

                        # Create an image file for each error
                        try:
                            plt.figure(figsize=(6, 6))

                            # Display the image based on its shape
                            if len(X_data.shape) == 4:  # For multi-channel images
                                if X_data.shape[3] == 1:  # Single channel
                                    plt.imshow(X_data[idx, :, :, 0], cmap='gray')
                                else:  # RGB or other
                                    plt.imshow(X_data[idx])
                            else:  # For other shapes
                                plt.imshow(X_data[idx])

                            plt.title(
                                f"True: {true_label}, Pred: {pred_label}\nProb: {prob:.3f}, Conf: {confidence:.3f}")
                            plt.axis('off')

                            # Create safe filenames
                            output_path = os.path.join(catalog_dir,
                                                       f"error_{i + 1}_true_{true_label}_pred_{pred_label}.png")
                            plt.savefig(output_path, dpi=150)
                            plt.close()

                            # Add this error to the data for CSV summary
                            error_data.append({
                                'Index': idx,
                                'True_Label': true_label,
                                'Predicted_Label': pred_label,
                                'Probability': prob,
                                'Confidence': confidence,
                                'Image_Path': f"error_{i + 1}_true_{true_label}_pred_{pred_label}.png"
                            })
                        except Exception as e:
                            logger.warning(f"Error saving example {i} (index {idx}): {e}")
                            plt.close()
                    except Exception as e:
                        logger.warning(f"Error processing example at index {idx}: {e}")
                        continue

                # Create a summary CSV
                try:
                    if error_data:
                        error_df = pd.DataFrame(error_data)
                        error_df.to_csv(os.path.join(catalog_dir, 'error_summary.csv'), index=False)
                        logger.info(f"Error catalog created with {len(error_data)} misclassified examples")
                    else:
                        logger.warning("No valid error examples to save in catalog")
                except Exception as e:
                    logger.warning(f"Error creating error summary CSV: {e}")
        except Exception as e:
            logger.warning(f"Error in _create_error_catalog: {e}")
            plt.close('all')  # Ensure all figures are closed

    def _analyze_decision_boundary(self, X_data, y_true, y_pred, y_pred_prob, boundary_indices):
        """
        Analyze examples near the decision boundary.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_prob: Prediction probabilities
            boundary_indices: Indices of examples near the decision boundary
        """
        try:
            if len(boundary_indices) == 0:
                logger.info("No examples found near the decision boundary")
                return

            # Create a directory for boundary analysis
            if self.output_dir:
                try:
                    boundary_dir = os.path.join(self.output_dir, 'error_analysis', 'decision_boundary')
                    os.makedirs(boundary_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not create decision boundary directory: {e}")
                    return

                # Plot distribution of prediction probabilities near boundary
                try:
                    plt.figure(figsize=(10, 6))
                    plt.hist(y_pred_prob[boundary_indices], bins=20, alpha=0.7)
                    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
                    plt.xlabel('Prediction Probability')
                    plt.ylabel('Count')
                    plt.title('Distribution of Prediction Probabilities Near Decision Boundary')
                    plt.legend()
                    plt.grid(alpha=0.3)

                    output_path = os.path.join(boundary_dir, 'boundary_probability_distribution.png')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Boundary probability distribution saved to {output_path}")
                    plt.close()
                except Exception as e:
                    logger.warning(f"Error plotting boundary probability distribution: {e}")
                    plt.close()

                # Create a summary of boundary examples
                try:
                    boundary_data = []
                    for i, idx in enumerate(boundary_indices):
                        boundary_data.append({
                            'Index': idx,
                            'True_Label': int(y_true[idx]),
                            'Predicted_Label': int(y_pred[idx]),
                            'Probability': float(y_pred_prob[idx]),
                            'Distance_to_Boundary': float(abs(y_pred_prob[idx] - 0.5)),
                            'Correct': y_true[idx] == y_pred[idx]
                        })

                    boundary_df = pd.DataFrame(boundary_data)
                    boundary_df.to_csv(os.path.join(boundary_dir, 'boundary_examples.csv'), index=False)
                    logger.info(
                        f"Boundary examples summary saved to {os.path.join(boundary_dir, 'boundary_examples.csv')}")
                except Exception as e:
                    logger.warning(f"Error creating boundary examples summary: {e}")

                # Visualize examples with GradCAM (wrapped in try/except to prevent failures)
                try:
                    self._visualize_boundary_examples_with_gradcam(X_data, boundary_indices, boundary_dir)
                except Exception as e:
                    logger.warning(f"Error visualizing boundary examples with GradCAM: {e}")
        except Exception as e:
            logger.warning(f"Error in _analyze_decision_boundary: {e}")

    def _visualize_boundary_examples_with_gradcam(self, X_data, boundary_indices, output_dir):
        """
        Visualize examples near decision boundary with GradCAM overlays.

        Args:
            X_data: Input data
            boundary_indices: Indices of examples near the decision boundary
            output_dir: Directory to save visualizations
        """
        try:
            # Limit to first few examples
            boundary_indices = boundary_indices[:min(5, len(boundary_indices))]

            # Check if we have a model to generate GradCAM
            if self.model is None:
                logger.warning("Model not available for GradCAM visualization")
                return

            try:
                # Import visualization if available
                from visualization import GradCAMGenerator

                # Find the last convolutional layer
                last_conv_layer = None
                for layer in reversed(self.model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer = layer.name
                        break

                if last_conv_layer is None:
                    logger.warning("No convolutional layer found for GradCAM")
                    return

                # Create GradCAM generator
                gradcam_gen = GradCAMGenerator(
                    model=self.model,
                    output_dir=None  # We'll handle saving manually
                )

                # Generate GradCAM for each example
                for i, idx in enumerate(boundary_indices):
                    try:
                        # Generate CAM
                        cam = gradcam_gen.compute_gradcam(X_data[idx], last_conv_layer)

                        # Create visualization
                        plt.figure(figsize=(12, 5))

                        # Original image
                        plt.subplot(1, 2, 1)
                        if len(X_data.shape) == 4 and X_data.shape[3] == 1:
                            plt.imshow(X_data[idx, :, :, 0], cmap='gray')
                        else:
                            plt.imshow(X_data[idx])
                        plt.title('Original Image')
                        plt.axis('off')

                        # GradCAM overlay
                        plt.subplot(1, 2, 2)
                        plt.imshow(cam, cmap='jet')
                        plt.title('GradCAM Activation')
                        plt.axis('off')

                        plt.suptitle(f'Boundary Example {i + 1}')
                        plt.tight_layout()

                        # Save visualization
                        output_path = os.path.join(output_dir, f'boundary_gradcam_{i + 1}.png')
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        logger.info(f"GradCAM visualization saved to {output_path}")
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Error generating GradCAM for example {i} (index {idx}): {e}")
                        plt.close()

            except ImportError:
                logger.info("GradCAMGenerator not available, skipping GradCAM visualization")

                # Define empty GradCAMGenerator class to avoid errors
                class GradCAMGenerator:
                    def __init__(self, **kwargs):
                        pass

                    def compute_gradcam(self, *args, **kwargs):
                        return np.zeros((10, 10))  # Return empty array
            except Exception as e:
                logger.warning(f"Error in GradCAM visualization: {e}")
                plt.close('all')
        except Exception as e:
            logger.warning(f"Error in _visualize_boundary_examples_with_gradcam: {e}")
            plt.close('all')  # Ensure all figures are closed

    def analyze_errors_by_family(self, X_test, y_test, family_labels):
        """
        Analyze errors grouped by malware family

        Args:
            X_test: Test data
            y_test: Test labels
            family_labels: Family labels for test samples

        Returns:
            Dictionary with family error analysis results
        """
        try:
            logger.info("Analyzing errors by malware family...")

            # Get model predictions
            try:
                y_pred_prob = self.model.predict(X_test)

                # Handle different prediction shapes
                if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
                    # For multi-class models, get the probability of the positive class
                    logger.info("Multi-dimensional output detected, using positive class probabilities")
                    y_pred_prob = y_pred_prob[:, 1]

                # Ensure y_pred_prob is flattened
                y_pred_prob = y_pred_prob.flatten()

                # Create the binary predictions using the threshold
                y_pred = (y_pred_prob > 0.5).astype(int)
                y_test = y_test.flatten()
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return {'error': f"Prediction failed: {str(e)}"}

            # Identify error indices
            error_indices = np.where(y_pred != y_test)[0]

            # Check if family_labels is the same length as y_test
            if len(family_labels) != len(y_test):
                logger.error(f"Length mismatch: family_labels ({len(family_labels)}) vs y_test ({len(y_test)})")
                return {
                    'error': 'Length mismatch between family labels and test labels',
                    'family_errors': {},
                    'family_error_rates': {},
                    'family_counts': {}
                }

            # Group errors by family
            family_errors = {}
            for idx in error_indices:
                try:
                    family = family_labels[idx]
                    if family not in family_errors:
                        family_errors[family] = []
                    family_errors[family].append(idx)
                except Exception as e:
                    logger.warning(f"Error processing error at index {idx}: {e}")
                    continue

            # Calculate error rates per family
            family_error_rates = {}
            family_counts = {}

            # Count all samples per family
            try:
                for i, family in enumerate(family_labels):
                    if family not in family_counts:
                        family_counts[family] = 0
                    family_counts[family] += 1
            except Exception as e:
                logger.warning(f"Error counting family samples: {e}")
                # Create empty counts in case of failure
                family_counts = {family: 0 for family in set(family_labels)}

            # Calculate error rates
            for family, errors in family_errors.items():
                try:
                    if family in family_counts and family_counts[family] > 0:
                        family_error_rates[family] = len(errors) / family_counts[family]
                    else:
                        family_error_rates[family] = 0
                except Exception as e:
                    logger.warning(f"Error calculating error rate for family {family}: {e}")
                    family_error_rates[family] = 0

            # Create visualization
            try:
                self._visualize_family_error_rates(family_error_rates, family_counts)
            except Exception as e:
                logger.warning(f"Error visualizing family error rates: {e}")

            return {
                'family_errors': family_errors,
                'family_error_rates': family_error_rates,
                'family_counts': family_counts
            }

        except Exception as e:
            logger.error(f"Error in analyze_errors_by_family: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'family_errors': {},
                'family_error_rates': {},
                'family_counts': {}
            }

    def _visualize_family_error_rates(self, family_error_rates, family_counts):
        """
        Visualize error rates by family

        Args:
            family_error_rates: Dictionary mapping family names to error rates
            family_counts: Dictionary mapping family names to sample counts
        """
        try:
            # Import needed libraries
            import matplotlib.pyplot as plt
            import pandas as pd
            from matplotlib.cm import viridis

            # Check if we have data to visualize
            if not family_error_rates or not family_counts:
                logger.warning("No family error rates or counts to visualize")
                return

            # Create data for visualization
            data = []
            for family, error_rate in family_error_rates.items():
                if family in family_counts:
                    data.append({
                        'Family': family,
                        'Error Rate': error_rate,
                        'Sample Count': family_counts[family]
                    })

            if not data:
                logger.warning("No valid family data to visualize")
                return

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Create directory for error analysis
            if self.output_dir:
                try:
                    error_dir = os.path.join(self.output_dir, 'error_analysis')
                    os.makedirs(error_dir, exist_ok=True)

                    # Save raw data
                    df.to_csv(os.path.join(error_dir, 'family_error_rates.csv'), index=False)
                    logger.info(f"Family error rates data saved to {os.path.join(error_dir, 'family_error_rates.csv')}")
                except Exception as e:
                    logger.warning(f"Error saving family error rates CSV: {e}")

            # Create error rate plot
            try:
                # Sort by sample count (descending)
                df = df.sort_values('Sample Count', ascending=False)

                # Create the plot
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(df['Family'], df['Error Rate'])

                # Color bars by sample count
                norm = plt.Normalize(df['Sample Count'].min(), df['Sample Count'].max())
                for i, bar in enumerate(bars):
                    bar.set_color(viridis(norm(df.iloc[i]['Sample Count'])))

                # Create colorbar with the specific axis reference
                sm = plt.cm.ScalarMappable(norm=norm, cmap=viridis)
                sm.set_array([])  # This line fixes the error
                fig.colorbar(sm, ax=ax, label='Sample Count')

                ax.set_title('Error Rate by Malware Family')
                ax.set_xlabel('Family')
                ax.set_ylabel('Error Rate')
                plt.xticks(rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()

                if self.output_dir:
                    output_path = os.path.join(error_dir, 'family_error_rates.png')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Family error rates plot saved to {output_path}")

                plt.close()

                # Additionally, create a scatter plot of error rate vs sample count
                plt.figure(figsize=(10, 8))
                plt.scatter(df['Sample Count'], df['Error Rate'],
                            s=100, alpha=0.7, c=range(len(df)), cmap='viridis')

                # Add family labels to points
                for i, row in df.iterrows():
                    plt.annotate(row['Family'],
                                 (row['Sample Count'], row['Error Rate']),
                                 xytext=(5, 5), textcoords='offset points')

                plt.xlabel('Sample Count')
                plt.ylabel('Error Rate')
                plt.title('Error Rate vs Sample Count by Family')
                plt.grid(alpha=0.3)

                if self.output_dir:
                    output_path = os.path.join(error_dir, 'error_rate_vs_sample_count.png')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Error rate vs sample count plot saved to {output_path}")

                plt.close()

            except Exception as e:
                logger.warning(f"Error creating family error rates visualization: {e}")
                plt.close('all')

        except Exception as e:
            logger.warning(f"Error in _visualize_family_error_rates: {e}")
            plt.close('all')  # Ensure all figures are closed