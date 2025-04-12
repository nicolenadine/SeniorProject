#!/usr/bin/env python3
"""
Comparative Metrics Module for Malware Classification System
Provides comprehensive performance metrics for model comparison
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    precision_score, recall_score, accuracy_score
)
import pandas as pd
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComparativeMetrics")


class ComparativeMetricsAnalyzer:
    def __init__(self, model=None, data_handler=None, output_dir=None, model_name="default"):
        """
        Initialize the comparative metrics analyzer.

        Args:
            model: Trained TensorFlow model
            data_handler: DataHandler instance containing the data
            output_dir: Directory to save metrics and visualizations
            model_name: Name or identifier for the model
        """
        self.model = model
        self.data_handler = data_handler
        self.output_dir = output_dir
        self.model_name = model_name

        if output_dir:
            try:
                os.makedirs(os.path.join(output_dir, 'comparative_metrics'), exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create output directory: {e}")

    def compute_metrics(self, X_data, y_true, compute_inference_time=True, n_thresholds=100):
        """
        Compute comprehensive performance metrics.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            compute_inference_time: Whether to compute inference time metrics
            n_thresholds: Number of thresholds to evaluate for optimal metrics

        Returns:
            Dictionary containing all computed metrics
        """
        logger.info("Computing comprehensive performance metrics...")

        # Initialize results dictionary
        results = {
            'model_name': self.model_name,
            'model_info': self._get_model_info(),
            'metrics': {}
        }

        try:
            # Make predictions
            start_time = time.time()
            y_pred_prob = self.model.predict(X_data)
            end_time = time.time()

            # Compute inference time metrics
            if compute_inference_time:
                total_time = end_time - start_time
                results['inference_time'] = {
                    'total_time_seconds': total_time,
                    'samples_per_second': len(X_data) / total_time,
                    'ms_per_sample': (total_time / len(X_data)) * 1000
                }

                # Compute per-batch inference time for more accurate measurements
                try:
                    batch_size = 32
                    n_samples = min(100, len(X_data))  # Limit to 100 samples for timing
                    X_subset = X_data[:n_samples]

                    batch_times = []
                    for i in range(0, n_samples, batch_size):
                        batch_X = X_subset[i:i + batch_size]
                        start_time = time.time()
                        self.model.predict(batch_X)
                        end_time = time.time()
                        batch_times.append(end_time - start_time)

                    # Calculate statistics
                    results['inference_time']['batch_stats'] = {
                        'mean_batch_time': float(np.mean(batch_times)),
                        'median_batch_time': float(np.median(batch_times)),
                        'std_batch_time': float(np.std(batch_times)),
                        'batch_size': batch_size
                    }
                except Exception as e:
                    logger.warning(f"Error computing batch inference time: {e}")
                    results['inference_time']['batch_stats'] = {
                        'error': str(e)
                    }

            # Ensure predictions are in the right shape for binary classification
            if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
                # For multi-class, use the positive class probability (assuming binary)
                logger.info("Multi-dimensional output detected, using positive class probabilities")
                y_pred_prob = y_pred_prob[:, 1]

            # Flatten predictions and labels
            y_pred_prob = y_pred_prob.flatten()
            y_true = y_true.flatten()

            # Compute standard threshold metrics (at 0.5)
            threshold = 0.5
            y_pred = (y_pred_prob >= threshold).astype(int)

            # Calculate basic metrics
            try:
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()

                # Store basic metrics
                results['metrics']['threshold_0.5'] = {
                    'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
                    'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
                    'recall': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
                    'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
                    'f1_score': float(f1_score(y_true, y_pred)),
                    'mcc': float(matthews_corrcoef(y_true, y_pred)),
                    'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
                    'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
                    'confusion_matrix': cm.tolist()
                }

                # Calculate F1 if true positives or predictions are non-zero
                if tp > 0 or (tp + fp) > 0:
                    results['metrics']['threshold_0.5']['f1_score'] = float(2 * tp / (2 * tp + fp + fn))
                else:
                    results['metrics']['threshold_0.5']['f1_score'] = 0.0
            except Exception as e:
                logger.warning(f"Error computing basic metrics: {e}")
                results['metrics']['threshold_0.5'] = {'error': str(e)}

            # Compute ROC curve metrics
            try:
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_prob)
                auc_value = float(roc_auc_score(y_true, y_pred_prob))
                results['roc'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist(),
                    'auc': auc_value
                }
            except Exception as e:
                logger.warning(f"Error computing ROC metrics: {e}")
                results['roc'] = {'error': str(e)}

            # Compute Precision-Recall curve metrics
            try:
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
                avg_precision = float(average_precision_score(y_true, y_pred_prob))
                results['precision_recall'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist(),
                    'average_precision': avg_precision
                }
            except Exception as e:
                logger.warning(f"Error computing Precision-Recall metrics: {e}")
                results['precision_recall'] = {'error': str(e)}

            # Find optimal threshold for various metrics
            try:
                results['optimal_thresholds'] = self._find_optimal_thresholds(y_true, y_pred_prob, n_thresholds)
            except Exception as e:
                logger.warning(f"Error finding optimal thresholds: {e}")
                results['optimal_thresholds'] = {'error': str(e)}

            # Store detailed classification report
            try:
                report = classification_report(y_true, y_pred, output_dict=True)
                results['classification_report'] = report
            except Exception as e:
                logger.warning(f"Error generating classification report: {e}")
                results['classification_report'] = {'error': str(e)}

            # Save results
            if self.output_dir:
                try:
                    # Save metrics as JSON
                    metrics_path = os.path.join(self.output_dir, 'comparative_metrics',
                                                f'{self.model_name}_metrics.json')
                    with open(metrics_path, 'w') as f:
                        json.dump(results, f, indent=4)
                    logger.info(f"Metrics saved to {metrics_path}")
                except Exception as e:
                    logger.warning(f"Error saving metrics to JSON: {e}")

            # Generate visualizations
            try:
                if 'roc' in results and 'error' not in results['roc']:
                    self._plot_roc_curve(results['roc']['fpr'], results['roc']['tpr'], results['roc']['auc'])
            except Exception as e:
                logger.warning(f"Error plotting ROC curve: {e}")

            try:
                if 'precision_recall' in results and 'error' not in results['precision_recall']:
                    self._plot_precision_recall_curve(
                        results['precision_recall']['precision'],
                        results['precision_recall']['recall'],
                        results['precision_recall']['average_precision']
                    )
            except Exception as e:
                logger.warning(f"Error plotting Precision-Recall curve: {e}")

            try:
                self._plot_threshold_metrics(y_true, y_pred_prob, n_thresholds)
            except Exception as e:
                logger.warning(f"Error plotting threshold metrics: {e}")

        except Exception as e:
            logger.error(f"Error in compute_metrics: {e}")
            results['error'] = str(e)

        return results

    def compare_with_baseline(self, baseline_metrics_path, current_metrics=None):
        """
        Compare current model metrics with a baseline model.

        Args:
            baseline_metrics_path: Path to the baseline model metrics JSON
            current_metrics: Current model metrics (if None, will be loaded from output_dir)

        Returns:
            Dictionary containing the comparison results
        """
        try:
            # Load baseline metrics
            try:
                with open(baseline_metrics_path, 'r') as f:
                    baseline_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading baseline metrics: {e}")
                return {'error': f"Could not load baseline metrics: {str(e)}"}

            # Load current metrics if not provided
            if current_metrics is None:
                if self.output_dir is None:
                    return {'error': "No output directory specified to load current metrics"}

                current_metrics_path = os.path.join(self.output_dir, 'comparative_metrics',
                                                    f'{self.model_name}_metrics.json')
                if not os.path.exists(current_metrics_path):
                    return {'error': f"Current metrics not found at {current_metrics_path}"}

                try:
                    with open(current_metrics_path, 'r') as f:
                        current_metrics = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading current metrics: {e}")
                    return {'error': f"Could not load current metrics: {str(e)}"}

            # Create comparison dictionary
            comparison = {
                'baseline_model': baseline_metrics.get('model_name', 'baseline'),
                'current_model': current_metrics.get('model_name', 'current'),
                'metrics_comparison': {},
                'inference_time_comparison': {}
            }

            # Compare standard metrics
            if 'metrics' in current_metrics and 'threshold_0.5' in current_metrics['metrics']:
                current_standard_metrics = current_metrics['metrics']['threshold_0.5']
                baseline_standard_metrics = baseline_metrics.get('metrics', {}).get('threshold_0.5', {})

                for metric, value in current_standard_metrics.items():
                    if metric != 'confusion_matrix':  # Skip confusion matrix from direct comparison
                        baseline_value = baseline_standard_metrics.get(metric, 0)
                        difference = value - baseline_value
                        percent_change = (difference / baseline_value) * 100 if baseline_value != 0 else float('inf')

                        comparison['metrics_comparison'][metric] = {
                            'baseline': baseline_value,
                            'current': value,
                            'absolute_difference': difference,
                            'percent_change': percent_change
                        }

            # Compare ROC AUC
            if 'roc' in current_metrics and 'auc' in current_metrics['roc']:
                current_auc = current_metrics['roc']['auc']
                baseline_auc = baseline_metrics.get('roc', {}).get('auc', 0)
                difference = current_auc - baseline_auc
                percent_change = (difference / baseline_auc) * 100 if baseline_auc != 0 else float('inf')

                comparison['metrics_comparison']['roc_auc'] = {
                    'baseline': baseline_auc,
                    'current': current_auc,
                    'absolute_difference': difference,
                    'percent_change': percent_change
                }

            # Compare AP (average precision)
            if 'precision_recall' in current_metrics and 'average_precision' in current_metrics['precision_recall']:
                current_ap = current_metrics['precision_recall']['average_precision']
                baseline_ap = baseline_metrics.get('precision_recall', {}).get('average_precision', 0)
                difference = current_ap - baseline_ap
                percent_change = (difference / baseline_ap) * 100 if baseline_ap != 0 else float('inf')

                comparison['metrics_comparison']['average_precision'] = {
                    'baseline': baseline_ap,
                    'current': current_ap,
                    'absolute_difference': difference,
                    'percent_change': percent_change
                }

            # Compare inference time if available
            if 'inference_time' in current_metrics and 'inference_time' in baseline_metrics:
                for metric, value in current_metrics['inference_time'].items():
                    if metric != 'batch_stats':  # Handle batch stats separately
                        baseline_value = baseline_metrics['inference_time'].get(metric, 0)
                        difference = value - baseline_value
                        percent_change = (difference / baseline_value) * 100 if baseline_value != 0 else float('inf')

                        comparison['inference_time_comparison'][metric] = {
                            'baseline': baseline_value,
                            'current': value,
                            'absolute_difference': difference,
                            'percent_change': percent_change
                        }

            # Compare model size and complexity
            current_params = current_metrics.get('model_info', {}).get('total_params', 0)
            baseline_params = baseline_metrics.get('model_info', {}).get('total_params', 0)

            comparison['model_comparison'] = {
                'baseline_params': baseline_params,
                'current_params': current_params,
                'param_difference': current_params - baseline_params,
                'param_ratio': current_params / baseline_params if baseline_params > 0 else float('inf')
            }

            # Save comparison
            if self.output_dir:
                try:
                    comparison_path = os.path.join(
                        self.output_dir,
                        'comparative_metrics',
                        f'comparison_{comparison["baseline_model"]}_vs_{comparison["current_model"]}.json'
                    )
                    with open(comparison_path, 'w') as f:
                        json.dump(comparison, f, indent=4)

                    logger.info(f"Comparison saved to {comparison_path}")
                except Exception as e:
                    logger.warning(f"Error saving comparison to JSON: {e}")

            # Create visualization of the comparison
            try:
                self._plot_metrics_comparison(comparison)
            except Exception as e:
                logger.warning(f"Error plotting metrics comparison: {e}")

            return comparison

        except Exception as e:
            logger.error(f"Error in compare_with_baseline: {e}")
            return {'error': str(e)}

    def _get_model_info(self):
        """
        Extract information about the model architecture.

        Returns:
            Dictionary containing model information
        """
        try:
            if self.model is None:
                return {
                    'total_params': 0,
                    'trainable_params': 0,
                    'non_trainable_params': 0
                }

            # Count parameters
            trainable_params = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
            non_trainable_params = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
            total_params = trainable_params + non_trainable_params

            # Get layer information
            layers_info = []
            for layer in self.model.layers:
                layer_info = {
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'output_shape': str(layer.output_shape),
                    'params': layer.count_params()
                }
                layers_info.append(layer_info)

            # Create model info dictionary
            model_info = {
                'model_name': self.model_name,
                'total_params': int(total_params),
                'trainable_params': int(trainable_params),
                'non_trainable_params': int(non_trainable_params),
                'layers': layers_info
            }

            return model_info

        except Exception as e:
            logger.warning(f"Error in _get_model_info: {e}")
            return {
                'error': str(e),
                'total_params': 0,
                'trainable_params': 0,
                'non_trainable_params': 0
            }

    def _find_optimal_thresholds(self, y_true, y_pred_prob, n_thresholds=100):
        """
        Find optimal classification thresholds for various metrics.

        Args:
            y_true: Ground truth labels
            y_pred_prob: Prediction probabilities
            n_thresholds: Number of thresholds to evaluate

        Returns:
            Dictionary containing optimal thresholds for different metrics
        """
        thresholds = np.linspace(0.01, 0.99, n_thresholds)

        # Initialize metrics
        accuracy = np.zeros(n_thresholds)
        precision = np.zeros(n_thresholds)
        recall = np.zeros(n_thresholds)
        specificity = np.zeros(n_thresholds)
        f1 = np.zeros(n_thresholds)
        mcc = np.zeros(n_thresholds)
        balanced_acc = np.zeros(n_thresholds)

        # Compute metrics for each threshold
        for i, threshold in enumerate(thresholds):
            y_pred = (y_pred_prob >= threshold).astype(int)

            try:
                # Calculate confusion matrix elements
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()

                # Calculate metrics
                accuracy[i] = (tp + tn) / (tp + tn + fp + fn)
                precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Calculate F1 if true positives or predictions are non-zero
                if tp > 0 or (tp + fp) > 0:
                    f1[i] = 2 * tp / (2 * tp + fp + fn)
                else:
                    f1[i] = 0.0

                mcc[i] = matthews_corrcoef(y_true, y_pred)
                balanced_acc[i] = balanced_accuracy_score(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Error calculating metrics for threshold {threshold}: {e}")
                # Set metrics to 0 for this threshold
                accuracy[i] = precision[i] = recall[i] = specificity[i] = f1[i] = mcc[i] = balanced_acc[i] = 0

        # Find optimal thresholds
        optimal_thresholds = {
            'accuracy': {
                'threshold': float(thresholds[np.argmax(accuracy)]),
                'value': float(np.max(accuracy))
            },
            'precision': {
                'threshold': float(thresholds[np.argmax(precision)]),
                'value': float(np.max(precision))
            },
            'recall': {
                'threshold': float(thresholds[np.argmax(recall)]),
                'value': float(np.max(recall))
            },
            'specificity': {
                'threshold': float(thresholds[np.argmax(specificity)]),
                'value': float(np.max(specificity))
            },
            'f1_score': {
                'threshold': float(thresholds[np.argmax(f1)]),
                'value': float(np.max(f1))
            },
            'mcc': {
                'threshold': float(thresholds[np.argmax(mcc)]),
                'value': float(np.max(mcc))
            },
            'balanced_accuracy': {
                'threshold': float(thresholds[np.argmax(balanced_acc)]),
                'value': float(np.max(balanced_acc))
            },
            # Youden's J statistic (sensitivity + specificity - 1)
            'youdens_j': {
                'threshold': float(thresholds[np.argmax(recall + specificity - 1)]),
                'value': float(np.max(recall + specificity - 1))
            }
        }

        # Add threshold curves for plotting
        optimal_thresholds['curves'] = {
            'thresholds': thresholds.tolist(),
            'accuracy': accuracy.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'specificity': specificity.tolist(),
            'f1_score': f1.tolist(),
            'mcc': mcc.tolist(),
            'balanced_accuracy': balanced_acc.tolist()
        }

        return optimal_thresholds

    def _plot_roc_curve(self, fpr, tpr, auc):
        """
        Plot ROC curve.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc: Area under the ROC curve
        """
        try:
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'comparative_metrics', f'{self.model_name}_roc_curve.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve saved to {output_path}")

            plt.close()
        except Exception as e:
            logger.warning(f"Error in _plot_roc_curve: {e}")

    def _plot_precision_recall_curve(self, precision, recall, average_precision):
        """
        Plot Precision-Recall curve.

        Args:
            precision: Precision values
            recall: Recall values
            average_precision: Average precision score
        """
        try:
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.4f})')
            plt.axhline(y=sum(precision) / len(precision), color='gray', lw=1, linestyle='--', label='Baseline')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.grid(alpha=0.3)

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'comparative_metrics', f'{self.model_name}_pr_curve.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Precision-Recall curve saved to {output_path}")

            plt.close()
        except Exception as e:
            logger.warning(f"Error in _plot_precision_recall_curve: {e}")

    def _plot_threshold_metrics(self, y_true, y_pred_prob, n_thresholds=100):
        """
        Plot metrics as a function of threshold.

        Args:
            y_true: Ground truth labels
            y_pred_prob: Prediction probabilities
            n_thresholds: Number of thresholds to evaluate
        """
        try:
            # Find optimal thresholds
            optimal_thresholds = self._find_optimal_thresholds(y_true, y_pred_prob, n_thresholds)
            if 'error' in optimal_thresholds:
                logger.warning(f"Error finding optimal thresholds: {optimal_thresholds['error']}")
                return

            curves = optimal_thresholds['curves']

            # Plot accuracy, precision, recall, and F1 score
            plt.figure(figsize=(12, 10))
            plt.plot(curves['thresholds'], curves['accuracy'], label='Accuracy', lw=2)
            plt.plot(curves['thresholds'], curves['precision'], label='Precision', lw=2)
            plt.plot(curves['thresholds'], curves['recall'], label='Recall', lw=2)
            plt.plot(curves['thresholds'], curves['f1_score'], label='F1 Score', lw=2)

            # Mark optimal thresholds
            plt.axvline(x=optimal_thresholds['f1_score']['threshold'], color='purple',
                        linestyle='--', alpha=0.7,
                        label=f'Optimal F1 Threshold = {optimal_thresholds["f1_score"]["threshold"]:.2f}')

            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold = 0.5')

            plt.xlabel('Threshold')
            plt.ylabel('Metric Value')
            plt.title('Metrics vs Classification Threshold')
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'comparative_metrics',
                                           f'{self.model_name}_threshold_metrics.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Threshold metrics plot saved to {output_path}")

            plt.close()

            # Plot balanced accuracy, MCC, and specificity
            plt.figure(figsize=(12, 10))
            plt.plot(curves['thresholds'], curves['balanced_accuracy'], label='Balanced Accuracy', lw=2)
            plt.plot(curves['thresholds'], curves['mcc'], label='Matthews Correlation Coefficient', lw=2)
            plt.plot(curves['thresholds'], curves['specificity'], label='Specificity', lw=2)

            # Mark optimal thresholds
            plt.axvline(x=optimal_thresholds['balanced_accuracy']['threshold'], color='green',
                        linestyle='--', alpha=0.7,
                        label=f'Optimal Balanced Acc Threshold = {optimal_thresholds["balanced_accuracy"]["threshold"]:.2f}')

            plt.axvline(x=optimal_thresholds['mcc']['threshold'], color='red',
                        linestyle='--', alpha=0.7,
                        label=f'Optimal MCC Threshold = {optimal_thresholds["mcc"]["threshold"]:.2f}')

            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold = 0.5')

            plt.xlabel('Threshold')
            plt.ylabel('Metric Value')
            plt.title('Advanced Metrics vs Classification Threshold')
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if self.output_dir:
                output_path = os.path.join(self.output_dir, 'comparative_metrics',
                                           f'{self.model_name}_advanced_threshold_metrics.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Advanced threshold metrics plot saved to {output_path}")

            plt.close()
        except Exception as e:
            logger.warning(f"Error in _plot_threshold_metrics: {e}")

    def _plot_metrics_comparison(self, comparison):
        """
        Plot comparison between current and baseline models.

        Args:
            comparison: Comparison dictionary
        """
        try:
            # Check if comparison has required data
            if 'metrics_comparison' not in comparison or not comparison['metrics_comparison']:
                logger.warning("No metrics comparison data available")
                return

            # Extract metric names and values
            metrics = []
            baseline_values = []
            current_values = []
            percent_changes = []

            # Process standard metrics
            for metric, values in comparison['metrics_comparison'].items():
                metrics.append(metric)
                baseline_values.append(values['baseline'])
                current_values.append(values['current'])
                percent_changes.append(values['percent_change'])

            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                'Metric': metrics,
                'Baseline': baseline_values,
                'Current': current_values,
                'Percent_Change': percent_changes
            })

            # Sort by percent change
            df = df.sort_values('Percent_Change', ascending=False)

            # Plot metric values side by side
            plt.figure(figsize=(14, 10))
            x = np.arange(len(df))
            width = 0.35

            plt.bar(x - width / 2, df['Baseline'], width, label=comparison['baseline_model'])
            plt.bar(x + width / 2, df['Current'], width, label=comparison['current_model'])

            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Performance Metrics Comparison')
            plt.xticks(x, df['Metric'], rotation=45, ha='right')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if self.output_dir:
                output_path = os.path.join(
                    self.output_dir,
                    'comparative_metrics',
                    f'comparison_{comparison["baseline_model"]}_vs_{comparison["current_model"]}_metrics.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Metrics comparison plot saved to {output_path}")

            plt.close()

            # Plot percent changes
            plt.figure(figsize=(14, 10))
            colors = ['green' if x > 0 else 'red' for x in df['Percent_Change']]
            plt.bar(df['Metric'], df['Percent_Change'], color=colors)

            plt.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            plt.xlabel('Metric')
            plt.ylabel('Percent Change (%)')
            plt.title(f'Percent Change: {comparison["current_model"]} vs {comparison["baseline_model"]}')
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if self.output_dir:
                output_path = os.path.join(
                    self.output_dir,
                    'comparative_metrics',
                    f'comparison_{comparison["baseline_model"]}_vs_{comparison["current_model"]}_percent_change.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Percent change plot saved to {output_path}")

            plt.close()

            # If inference time comparison is available, plot it
            if comparison['inference_time_comparison']:
                # Extract inference time metrics
                time_metrics = []
                baseline_times = []
                current_times = []
                time_percent_changes = []

                for metric, values in comparison['inference_time_comparison'].items():
                    time_metrics.append(metric)
                    baseline_times.append(values['baseline'])
                    current_times.append(values['current'])
                    time_percent_changes.append(values['percent_change'])

                # Create DataFrame
                time_df = pd.DataFrame({
                    'Metric': time_metrics,
                    'Baseline': baseline_times,
                    'Current': current_times,
                    'Percent_Change': time_percent_changes
                })

                # Plot inference time comparison
                plt.figure(figsize=(12, 8))
                x = np.arange(len(time_df))
                width = 0.35

                plt.bar(x - width / 2, time_df['Baseline'], width, label=comparison['baseline_model'])
                plt.bar(x + width / 2, time_df['Current'], width, label=comparison['current_model'])

                plt.xlabel('Metric')
                plt.ylabel('Time (seconds)')
                plt.title('Inference Time Comparison')
                plt.xticks(x, time_df['Metric'], rotation=45, ha='right')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()

                if self.output_dir:
                    output_path = os.path.join(
                        self.output_dir,
                        'comparative_metrics',
                        f'comparison_{comparison["baseline_model"]}_vs_{comparison["current_model"]}_inference_time.png'
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Inference time comparison plot saved to {output_path}")

                plt.close()
        except Exception as e:
            logger.warning(f"Error in _plot_metrics_comparison: {e}")

    def measure_inference_time(self, X_data, batch_sizes=[1, 4, 8, 16, 32, 64, 128], n_repeats=5):
        """
        Measure inference time for different batch sizes.

        Args:
            X_data: Input data
            batch_sizes: List of batch sizes to test
            n_repeats: Number of repeats for each batch size

        Returns:
            Dictionary containing inference time measurements
        """
        if self.model is None:
            logger.warning("No model available for inference time measurement")
            return {'error': 'No model available'}

        try:
            # Initialize results
            results = {
                'model_name': self.model_name,
                'batch_sizes': batch_sizes,
                'times_per_batch': [],
                'times_per_sample': []
            }

            # Measure for each batch size
            for batch_size in batch_sizes:
                try:
                    # Limit to available samples
                    n_samples = min(batch_size * n_repeats, len(X_data))
                    n_batches = n_samples // batch_size

                    batch_times = []

                    # Warm up the model
                    warmup_batch = X_data[:batch_size]
                    for _ in range(3):
                        self.model.predict(warmup_batch)

                    # Measure inference time
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = start_idx + batch_size
                        batch_X = X_data[start_idx:end_idx]

                        # Time the prediction
                        start_time = time.time()
                        self.model.predict(batch_X)
                        end_time = time.time()

                        batch_times.append(end_time - start_time)

                    # Calculate statistics
                    mean_time = float(np.mean(batch_times))
                    time_per_sample = mean_time / batch_size

                    results['times_per_batch'].append(mean_time)
                    results['times_per_sample'].append(time_per_sample)

                except Exception as e:
                    logger.warning(f"Error measuring inference time for batch size {batch_size}: {e}")
                    results['times_per_batch'].append(None)
                    results['times_per_sample'].append(None)

            # Plot inference time results
            try:
                self._plot_inference_time_results(results)
            except Exception as e:
                logger.warning(f"Error plotting inference time results: {e}")

            # Save results
            if self.output_dir:
                try:
                    results_path = os.path.join(
                        self.output_dir,
                        'comparative_metrics',
                        f'{self.model_name}_inference_time.json'
                    )
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=4)
                    logger.info(f"Inference time results saved to {results_path}")
                except Exception as e:
                    logger.warning(f"Error saving inference time results: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in measure_inference_time: {e}")
            return {'error': str(e)}

    def _plot_inference_time_results(self, results):
        """
        Plot inference time results.

        Args:
            results: Inference time results dictionary
        """
        try:
            # Filter out None values
            valid_indices = [i for i, val in enumerate(results['times_per_batch']) if val is not None]
            valid_batch_sizes = [results['batch_sizes'][i] for i in valid_indices]
            valid_times_per_batch = [results['times_per_batch'][i] for i in valid_indices]
            valid_times_per_sample = [results['times_per_sample'][i] for i in valid_indices]

            if not valid_indices:
                logger.warning("No valid inference time data to plot")
                return

            # Plot time per batch
            plt.figure(figsize=(12, 8))
            plt.plot(valid_batch_sizes, valid_times_per_batch, 'o-', lw=2)
            plt.xlabel('Batch Size')
            plt.ylabel('Time per Batch (seconds)')
            plt.title(f'Inference Time per Batch - {self.model_name}')
            plt.grid(alpha=0.3)
            plt.xscale('log', base=2)
            plt.xticks(valid_batch_sizes, [str(x) for x in valid_batch_sizes])

            if self.output_dir:
                output_path = os.path.join(
                    self.output_dir,
                    'comparative_metrics',
                    f'{self.model_name}_inference_time_per_batch.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Inference time per batch plot saved to {output_path}")

            plt.close()

            # Plot time per sample
            plt.figure(figsize=(12, 8))
            plt.plot(valid_batch_sizes, valid_times_per_sample, 'o-', lw=2)
            plt.xlabel('Batch Size')
            plt.ylabel('Time per Sample (seconds)')
            plt.title(f'Inference Time per Sample - {self.model_name}')
            plt.grid(alpha=0.3)
            plt.xscale('log', base=2)
            plt.xticks(valid_batch_sizes, [str(x) for x in valid_batch_sizes])

            if self.output_dir:
                output_path = os.path.join(
                    self.output_dir,
                    'comparative_metrics',
                    f'{self.model_name}_inference_time_per_sample.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Inference time per sample plot saved to {output_path}")

            plt.close()
        except Exception as e:
            logger.warning(f"Error in _plot_inference_time_results: {e}")

    def generate_per_family_metrics(self, X_test, y_test, family_labels):
        """
        Generate metrics broken down by malware family

        Args:
            X_test: Test data
            y_test: Test labels
            family_labels: Family labels for test samples

        Returns:
            Dictionary with per-family metrics
        """
        logger.info("Generating per-family performance metrics...")

        # Initialize results
        results = {
            'per_family_metrics': {},
            'family_confusion': {},
            'overall_metrics': {}  # Add overall metrics for comparison
        }

        try:
            # Get model predictions
            y_pred_proba = self.model.predict(X_test)

            # Ensure predictions are in the right shape for binary classification
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]

            y_pred_proba = y_pred_proba.flatten()
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_test = y_test.flatten()

            # Calculate overall metrics for comparison
            try:
                overall_accuracy = accuracy_score(y_test, y_pred)
                overall_precision = precision_score(y_test, y_pred, zero_division=0)
                overall_recall = recall_score(y_test, y_pred, zero_division=0)
                overall_f1 = f1_score(y_test, y_pred, zero_division=0)
                overall_auc = roc_auc_score(y_test, y_pred_proba)

                results['overall_metrics'] = {
                    'accuracy': float(overall_accuracy),
                    'precision': float(overall_precision),
                    'recall': float(overall_recall),
                    'f1_score': float(overall_f1),
                    'auc': float(overall_auc)
                }
            except Exception as e:
                logger.warning(f"Error computing overall metrics: {e}")
                results['overall_metrics'] = {'error': str(e)}

            # Get unique families
            unique_families = list(set(family_labels))

            # For each family, compute metrics
            for family in unique_families:
                try:
                    # Get indices for this family
                    family_indices = [i for i, f in enumerate(family_labels) if f == family]

                    if not family_indices:
                        continue

                    # Extract predictions and ground truth for this family
                    family_y_true = np.array([y_test[i] for i in family_indices])
                    family_y_pred = np.array([y_pred[i] for i in family_indices])
                    family_y_pred_proba = np.array([y_pred_proba[i] for i in family_indices])

                    # Skip if all samples are from the same class (avoid division by zero in AUC)
                    if len(set(family_y_true)) < 2:
                        logger.info(f"Skipping family {family}: all samples have the same label")
                        continue

                    # Compute metrics
                    accuracy = accuracy_score(family_y_true, family_y_pred)
                    precision = precision_score(family_y_true, family_y_pred, zero_division=0)
                    recall = recall_score(family_y_true, family_y_pred, zero_division=0)
                    f1 = f1_score(family_y_true, family_y_pred, zero_division=0)

                    # Compute AUC if possible
                    try:
                        auc = roc_auc_score(family_y_true, family_y_pred_proba)
                    except Exception as e:
                        logger.warning(f"Error computing AUC for family {family}: {e}")
                        auc = None

                    results['per_family_metrics'][family] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'sample_count': len(family_indices)
                    }

                    # Add AUC if available
                    if auc is not None:
                        results['per_family_metrics'][family]['auc'] = float(auc)

                except Exception as e:
                    logger.warning(f"Error computing metrics for family {family}: {e}")
                    results['per_family_metrics'][family] = {
                        'error': str(e),
                        'sample_count': len([i for i, f in enumerate(family_labels) if f == family])
                    }

            # Create directory for visualizations
            if self.output_dir:
                try:
                    vis_dir = os.path.join(self.output_dir, 'family_metrics')
                    os.makedirs(vis_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Error creating family metrics directory: {e}")

            # Create visualizations
            try:
                self._create_family_performance_visualizations(results['per_family_metrics'])
            except Exception as e:
                logger.warning(f"Error creating family performance visualizations: {e}")

            # If we have overall metrics and family metrics with AUC, plot the comparison
            if 'overall_metrics' in results and results['overall_metrics'] and 'auc' in results['overall_metrics']:
                try:
                    self._plot_per_family_metrics(results)
                except Exception as e:
                    logger.warning(f"Error plotting per-family metrics comparison: {e}")

            # Save results
            if self.output_dir:
                try:
                    results_path = os.path.join(
                        self.output_dir,
                        'family_metrics',
                        f'{self.model_name}_family_metrics.json'
                    )
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=4)
                    logger.info(f"Family metrics saved to {results_path}")
                except Exception as e:
                    logger.warning(f"Error saving family metrics: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in generate_per_family_metrics: {e}")
            return {
                'error': str(e),
                'per_family_metrics': {},
                'family_confusion': {},
                'overall_metrics': {}
            }

    def _create_family_performance_visualizations(self, family_metrics):
        """
        Create visualizations for family-level metrics

        Args:
            family_metrics: Dictionary with family metrics
        """
        try:
            if not family_metrics:
                logger.warning("No family metrics to visualize")
                return

            # Create directory for visualizations
            if self.output_dir:
                vis_dir = os.path.join(self.output_dir, 'family_metrics')
                os.makedirs(vis_dir, exist_ok=True)

            # Convert to DataFrame for easier plotting
            data = []
            for family, metrics in family_metrics.items():
                if 'error' in metrics:
                    continue

                row = {'Family': family}
                row.update(metrics)
                data.append(row)

            if not data:
                logger.warning("No valid family metrics to visualize")
                return

            df = pd.DataFrame(data)

            # Save raw data
            if self.output_dir:
                try:
                    df.to_csv(os.path.join(vis_dir, 'family_metrics.csv'), index=False)
                    logger.info(f"Family metrics CSV saved")
                except Exception as e:
                    logger.warning(f"Error saving family metrics CSV: {e}")

            # Create bar chart of F1 scores
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Family', y='f1_score', data=df)
            plt.title('F1 Score by Malware Family')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if self.output_dir:
                try:
                    plt.savefig(os.path.join(vis_dir, 'family_f1_scores.png'), dpi=300)
                    logger.info(f"Family F1 scores plot saved")
                except Exception as e:
                    logger.warning(f"Error saving F1 scores plot: {e}")

            plt.close()

            # Create heatmap of all metrics
            metrics_df = df.set_index('Family')
            # Remove non-metric columns
            for col in ['sample_count', 'error']:
                if col in metrics_df.columns:
                    metrics_df = metrics_df.drop(columns=[col])

            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
            plt.title('Performance Metrics by Malware Family')
            plt.tight_layout()

            if self.output_dir:
                try:
                    plt.savefig(os.path.join(vis_dir, 'family_metrics_heatmap.png'), dpi=300)
                    logger.info(f"Family metrics heatmap saved")
                except Exception as e:
                    logger.warning(f"Error saving metrics heatmap: {e}")

            plt.close()

            # Create sample count visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Family', y='sample_count', data=df)
            plt.title('Sample Count by Malware Family')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if self.output_dir:
                try:
                    plt.savefig(os.path.join(vis_dir, 'family_sample_counts.png'), dpi=300)
                    logger.info(f"Family sample counts plot saved")
                except Exception as e:
                    logger.warning(f"Error saving sample counts plot: {e}")

            plt.close()

        except Exception as e:
            logger.warning(f"Error in _create_family_performance_visualizations: {e}")

    def _plot_per_family_metrics(self, results):
        """
        Plot per-family metrics compared to overall performance.

        Args:
            results: Per-family metrics results dictionary
        """
        try:
            if 'per_family_metrics' not in results or not results['per_family_metrics']:
                logger.warning("No per-family metrics to plot")
                return

            if 'overall_metrics' not in results or not results['overall_metrics']:
                logger.warning("No overall metrics available for comparison")
                return

            # Extract family names and metrics
            families = []
            sample_counts = []
            accuracy = []
            precision = []
            recall = []
            f1 = []
            auc = []

            # Flag to check if auc is available
            has_auc = True

            for family, metrics in results['per_family_metrics'].items():
                if 'error' in metrics:
                    continue

                families.append(family)
                sample_counts.append(metrics.get('sample_count', 0))
                accuracy.append(metrics.get('accuracy', 0))
                precision.append(metrics.get('precision', 0))
                recall.append(metrics.get('recall', 0))
                f1.append(metrics.get('f1_score', 0))

                # Check if AUC is available
                if 'auc' in metrics:
                    auc.append(metrics.get('auc', 0))
                else:
                    has_auc = False

            if not families:
                logger.warning("No valid family metrics found")
                return

            # Sort families by sample count (descending)
            sort_indices = np.argsort(sample_counts)[::-1]
            families = [families[i] for i in sort_indices]
            sample_counts = [sample_counts[i] for i in sort_indices]
            accuracy = [accuracy[i] for i in sort_indices]
            precision = [precision[i] for i in sort_indices]
            recall = [recall[i] for i in sort_indices]
            f1 = [f1[i] for i in sort_indices]

            if has_auc:
                auc = [auc[i] for i in sort_indices]

            # Plot metrics for each family
            plt.figure(figsize=(14, 10))
            x = np.arange(len(families))
            width = 0.15

            plt.bar(x - 2 * width, accuracy, width, label='Accuracy')
            plt.bar(x - width, precision, width, label='Precision')
            plt.bar(x, recall, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1 Score')

            if has_auc:
                plt.bar(x + 2 * width, auc, width, label='AUC')

            # Add overall metrics as horizontal lines
            overall = results['overall_metrics']
            plt.axhline(y=overall.get('accuracy', 0), color='blue', linestyle='--', alpha=0.5, label='Overall Accuracy')
            plt.axhline(y=overall.get('precision', 0), color='orange', linestyle='--', alpha=0.5,
                        label='Overall Precision')
            plt.axhline(y=overall.get('recall', 0), color='green', linestyle='--', alpha=0.5, label='Overall Recall')
            plt.axhline(y=overall.get('f1_score', 0), color='red', linestyle='--', alpha=0.5, label='Overall F1')

            if has_auc and 'auc' in overall:
                plt.axhline(y=overall.get('auc', 0), color='purple', linestyle='--', alpha=0.5, label='Overall AUC')

            plt.xlabel('Malware Family')
            plt.ylabel('Metric Value')
            plt.title(f'Performance Metrics by Malware Family - {self.model_name}')
            plt.xticks(x, families, rotation=45, ha='right')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            if self.output_dir:
                output_path = os.path.join(
                    self.output_dir,
                    'family_metrics',
                    f'{self.model_name}_per_family_metrics.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Per-family metrics comparison plot saved to {output_path}")

            plt.close()

            # Create a comparison with overall performance
            # Find the best and worst performing families for each metric
            best_families = {
                'accuracy': families[np.argmax(accuracy)],
                'precision': families[np.argmax(precision)],
                'recall': families[np.argmax(recall)],
                'f1_score': families[np.argmax(f1)]
            }

            if has_auc:
                best_families['auc'] = families[np.argmax(auc)]

            worst_families = {
                'accuracy': families[np.argmin(accuracy)],
                'precision': families[np.argmin(precision)],
                'recall': families[np.argmin(recall)],
                'f1_score': families[np.argmin(f1)]
            }

            if has_auc:
                worst_families['auc'] = families[np.argmin(auc)]

            # Create a DataFrame for the comparison
            comparison_data = []
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
            if has_auc:
                metrics_to_compare.append('auc')

            for metric in metrics_to_compare:
                best_value = max([results['per_family_metrics'][fam].get(metric, 0)
                                  for fam in families if 'error' not in results['per_family_metrics'][fam]])
                worst_value = min([results['per_family_metrics'][fam].get(metric, 0)
                                   for fam in families if 'error' not in results['per_family_metrics'][fam]])

                comparison_data.append({
                    'Metric': metric,
                    'Overall': overall.get(metric, 0),
                    'Best Family': best_families.get(metric, 'Unknown'),
                    'Best Value': best_value,
                    'Worst Family': worst_families.get(metric, 'Unknown'),
                    'Worst Value': worst_value
                })

            comparison_df = pd.DataFrame(comparison_data)

            # Save comparison table
            if self.output_dir:
                try:
                    table_path = os.path.join(
                        self.output_dir,
                        'family_metrics',
                        f'{self.model_name}_per_family_comparison.csv'
                    )
                    comparison_df.to_csv(table_path, index=False)
                    logger.info(f"Per-family comparison table saved to {table_path}")
                except Exception as e:
                    logger.warning(f"Error saving per-family comparison table: {e}")

        except Exception as e:
            logger.warning(f"Error in _plot_per_family_metrics: {e}")