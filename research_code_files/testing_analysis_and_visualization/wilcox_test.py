#!/usr/bin/env python3
"""
Statistical Comparison Module for Malware Classification System

This module implements the Wilcoxon signed-rank test to compare the performance of
two classification models (full image CNN vs. segmented CNN) using the same test samples.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc


class StatisticalComparison:
    def __init__(self, full_model_dir, segment_model_dir, output_dir=None):
        """
        Initialize the statistical comparison module.

        Args:
            full_model_dir: Directory containing the full image model results
            segment_model_dir: Directory containing the segment model results
            output_dir: Directory to save the comparison results (default: 'results/comparison')
        """
        self.full_model_dir = full_model_dir
        self.segment_model_dir = segment_model_dir

        # Create output directory if not provided
        if output_dir is None:
            output_dir = 'results/statistical_comparison'
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # These will be populated when loading data
        self.test_files = None
        self.test_labels = None
        self.full_model_probs = None
        self.segment_model_probs = None
        self.full_model_preds = None
        self.segment_model_preds = None
        self.family_labels = None

    def load_test_data(self):
        """
        Load the test data, model predictions, and ground truth labels.

        Returns:
            bool: True if data loading was successful, False otherwise
        """
        print("Loading test data and model predictions...")

        # Get test files and ground truth labels
        try:
            # First try to get test files from data_splits directory
            splits_dir = os.path.join(self.full_model_dir, 'data_splits')

            with open(os.path.join(splits_dir, 'test_files.txt')) as f:
                self.test_files = f.read().splitlines()

            with open(os.path.join(splits_dir, 'test_labels.txt')) as f:
                self.test_labels = np.array([int(x) for x in f.read().splitlines()])

            # Try to load family labels if available
            try:
                with open(os.path.join(splits_dir, 'family_labels.txt')) as f:
                    all_family_labels = f.read().splitlines()

                # Extract just the test set family labels using the indices
                # This assumes test files are in the same order as in the original dataset
                test_indices = []
                for test_file in self.test_files:
                    # Find the index of this file in the original data
                    for i, file_path in enumerate(self.test_files):
                        if file_path == test_file:
                            test_indices.append(i)
                            break

                if test_indices:
                    self.family_labels = [all_family_labels[i] for i in test_indices]
            except Exception as e:
                print(f"Note: Could not load family labels. Will proceed without family-level analysis. Error: {e}")
                self.family_labels = None

        except Exception as e:
            print(f"Could not load test files from data_splits directory: {e}")
            print("Looking for numpy arrays instead...")

            # Try to load from numpy arrays if the text files don't exist
            try:
                metrics_dir = os.path.join(self.full_model_dir, 'metrics')
                self.test_files = np.load(os.path.join(metrics_dir, 'test_file_paths.npy'))
                self.test_labels = np.load(os.path.join(metrics_dir, 'y_true.npy'))
            except Exception as e2:
                print(f"Error loading test files and labels: {e2}")
                return False

        # Load full model predictions
        try:
            # First try to load from metrics dir
            metrics_dir = os.path.join(self.full_model_dir, 'metrics')

            if os.path.exists(os.path.join(metrics_dir, 'y_pred_prob.npy')):
                self.full_model_probs = np.load(os.path.join(metrics_dir, 'y_pred_prob.npy')).flatten()
                self.full_model_preds = np.load(os.path.join(metrics_dir, 'y_pred.npy')).flatten()
            else:
                # Try CSV format
                pred_csv = os.path.join(metrics_dir, 'test_predictions.csv')
                if os.path.exists(pred_csv):
                    df = pd.read_csv(pred_csv)
                    self.full_model_probs = df['prediction_probability'].values
                    self.full_model_preds = df['predicted_label'].values
                else:
                    print("Could not find full model predictions in expected formats")
                    return False
        except Exception as e:
            print(f"Error loading full model predictions: {e}")
            return False

        # Load segment model predictions
        try:
            # First try NPY format
            self.segment_model_probs = np.load(os.path.join(self.segment_model_dir, 'test_probabilities.npy')).flatten()
            self.segment_model_preds = np.load(os.path.join(self.segment_model_dir, 'test_predictions.npy')).flatten()
        except Exception as e:
            print(f"Error loading segment model predictions: {e}")
            return False

        # Verify that we have all required data
        if (self.test_labels is None or self.full_model_probs is None or
                self.segment_model_probs is None or self.full_model_preds is None or
                self.segment_model_preds is None):
            print("Missing required data for comparison")
            return False

        # Verify data shapes
        if (len(self.test_labels) != len(self.full_model_probs) or
                len(self.test_labels) != len(self.segment_model_probs)):
            print(f"Data shape mismatch: test_labels: {len(self.test_labels)}, "
                  f"full_model_probs: {len(self.full_model_probs)}, "
                  f"segment_model_probs: {len(self.segment_model_probs)}")
            return False

        print(f"Successfully loaded data for {len(self.test_labels)} test samples")
        return True

    def calculate_paired_metrics(self):
        """
        Calculate per-sample performance metrics for both models.

        Returns:
            dict: Dictionary containing per-sample metrics for both models
        """
        print("Calculating per-sample metrics...")

        # Initialize a dictionary to store per-sample metrics
        metrics = {
            'sample_idx': [],
            'file_path': [],
            'true_label': [],
            'family': [] if self.family_labels is not None else None,
            'full_model': {
                'prediction': [],
                'probability': [],
                'correct': [],
                'log_loss': []
            },
            'segment_model': {
                'prediction': [],
                'probability': [],
                'correct': [],
                'log_loss': []
            },
            'performance_diff': {
                'prediction_match': [],
                'probability_diff': [],
                'log_loss_diff': []
            }
        }

        # Calculate metrics for each sample
        for i, (true_label, full_prob, segment_prob) in enumerate(
                zip(self.test_labels, self.full_model_probs, self.segment_model_probs)
        ):
            # Get file path
            file_path = self.test_files[i] if i < len(self.test_files) else f"sample_{i}"

            # Get predictions
            full_pred = self.full_model_preds[i]
            segment_pred = self.segment_model_preds[i]

            # Calculate log loss (capped to prevent infinite values)
            epsilon = 1e-15
            full_prob_capped = np.clip(full_prob, epsilon, 1 - epsilon)
            segment_prob_capped = np.clip(segment_prob, epsilon, 1 - epsilon)

            if true_label == 1:
                full_log_loss = -np.log(full_prob_capped)
                segment_log_loss = -np.log(segment_prob_capped)
            else:
                full_log_loss = -np.log(1 - full_prob_capped)
                segment_log_loss = -np.log(1 - segment_prob_capped)

            # Add to metrics dictionary
            metrics['sample_idx'].append(i)
            metrics['file_path'].append(file_path)
            metrics['true_label'].append(true_label)

            if self.family_labels is not None:
                if i < len(self.family_labels):
                    metrics['family'].append(self.family_labels[i])
                else:
                    metrics['family'].append("unknown")

            metrics['full_model']['prediction'].append(full_pred)
            metrics['full_model']['probability'].append(full_prob)
            metrics['full_model']['correct'].append(full_pred == true_label)
            metrics['full_model']['log_loss'].append(full_log_loss)

            metrics['segment_model']['prediction'].append(segment_pred)
            metrics['segment_model']['probability'].append(segment_prob)
            metrics['segment_model']['correct'].append(segment_pred == true_label)
            metrics['segment_model']['log_loss'].append(segment_log_loss)

            metrics['performance_diff']['prediction_match'].append(full_pred == segment_pred)
            metrics['performance_diff']['probability_diff'].append(full_prob - segment_prob)
            metrics['performance_diff']['log_loss_diff'].append(full_log_loss - segment_log_loss)

        return metrics

    def run_wilcoxon_test(self, metrics, alpha=0.05):
        """
        Run the Wilcoxon signed-rank test on the paired metrics.

        Args:
            metrics: Dictionary containing per-sample metrics
            alpha: Significance level (default: 0.05)

        Returns:
            dict: Dictionary containing the test results
        """
        print("Running Wilcoxon signed-rank test...")
        from scipy import stats

        # Extract metrics
        full_correct = np.array(metrics['full_model']['correct']).astype(int)
        segment_correct = np.array(metrics['segment_model']['correct']).astype(int)
        full_log_loss = np.array(metrics['full_model']['log_loss'])
        segment_log_loss = np.array(metrics['segment_model']['log_loss'])

        # Compute test on prediction correctness
        try:
            wilcoxon_result_correct = stats.wilcoxon(full_correct, segment_correct)
            p_value_correct = wilcoxon_result_correct.pvalue
            is_significant_correct = p_value_correct < alpha
            statistic_correct = wilcoxon_result_correct.statistic
        except Exception as e:
            print(f"Error computing Wilcoxon test on correctness: {e}")
            p_value_correct = None
            is_significant_correct = False
            statistic_correct = None

        # Compute test on log loss
        try:
            wilcoxon_result_loss = stats.wilcoxon(full_log_loss, segment_log_loss)
            p_value_loss = wilcoxon_result_loss.pvalue
            is_significant_loss = p_value_loss < alpha
            statistic_loss = wilcoxon_result_loss.statistic
        except Exception as e:
            print(f"Error computing Wilcoxon test on log loss: {e}")
            p_value_loss = None
            is_significant_loss = False
            statistic_loss = None

        # Determine which model is better
        full_model_acc = np.mean(full_correct)
        segment_model_acc = np.mean(segment_correct)
        full_model_loss = np.mean(full_log_loss)
        segment_model_loss = np.mean(segment_log_loss)

        better_model_acc = "Segment Model" if segment_model_acc > full_model_acc else "Full Model"
        better_model_loss = "Segment Model" if segment_model_loss < full_model_loss else "Full Model"

        # Create results dictionary
        results = {
            "wilcoxon_test_accuracy": {
                "statistic": float(statistic_correct) if statistic_correct is not None else None,
                "p_value": float(p_value_correct) if p_value_correct is not None else None,
                "is_significant": bool(is_significant_correct),
                "better_model": better_model_acc,
                "full_model_accuracy": float(full_model_acc),
                "segment_model_accuracy": float(segment_model_acc),
                "accuracy_difference": float(segment_model_acc - full_model_acc)
            },
            "wilcoxon_test_log_loss": {
                "statistic": float(statistic_loss) if statistic_loss is not None else None,
                "p_value": float(p_value_loss) if p_value_loss is not None else None,
                "is_significant": bool(is_significant_loss),
                "better_model": better_model_loss,
                "full_model_log_loss": float(full_model_loss),
                "segment_model_log_loss": float(segment_model_loss),
                "log_loss_difference": float(segment_model_loss - full_model_loss)
            }
        }

        return results

    def compute_mcnemar_test(self, contingency_table):
        """
        Manually implement McNemar's test since the function location varies by scipy version

        Args:
            contingency_table: 2x2 array with format:
                [both_correct, model1_correct_only]
                [model2_correct_only, both_incorrect]

        Returns:
            dict with test results
        """
        # Make sure contingency table is a numpy array
        if not isinstance(contingency_table, np.ndarray):
            contingency_table = np.array(contingency_table)

        # Extract counts
        b = contingency_table[0, 1]  # only model 1 correct
        c = contingency_table[1, 0]  # only model 2 correct

        # Special case: if b + c = 0, then the test isn't applicable
        if b + c == 0:
            return {
                "contingency_table": contingency_table.tolist(),
                "p_value": 1.0,
                "is_significant": False,
                "statistic": 0.0,
                "better_model": "Equal"
            }

        # Calculate test statistic with continuity correction
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)

        # Calculate p-value from chi-square distribution with df=1
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        is_significant = p_value < 0.05

        # Determine better model
        better_model = "Segment Model" if c > b else "Full Model" if b > c else "Equal"

        return {
            "contingency_table": contingency_table.tolist(),
            "p_value": float(p_value),
            "is_significant": bool(is_significant),
            "statistic": float(statistic),
            "better_model": better_model
        }

    def compute_aggregate_metrics(self, metrics):
        """
        Compute aggregate metrics for both models on the test set.

        Args:
            metrics: Dictionary containing per-sample metrics

        Returns:
            dict: Dictionary containing aggregate metrics
        """
        print("Computing aggregate performance metrics...")

        # Extract data
        y_true = np.array(metrics['true_label'])
        full_preds = np.array(metrics['full_model']['prediction'])
        segment_preds = np.array(metrics['segment_model']['prediction'])
        full_probs = np.array(metrics['full_model']['probability'])
        segment_probs = np.array(metrics['segment_model']['probability'])

        # Compute metrics for full model
        full_model_metrics = {
            "accuracy": float(accuracy_score(y_true, full_preds)),
            "precision": float(precision_score(y_true, full_preds, zero_division=0)),
            "recall": float(recall_score(y_true, full_preds, zero_division=0)),
            "f1_score": float(f1_score(y_true, full_preds, zero_division=0)),
            "auc": float(roc_auc_score(y_true, full_probs))
        }

        # Compute metrics for segment model
        segment_model_metrics = {
            "accuracy": float(accuracy_score(y_true, segment_preds)),
            "precision": float(precision_score(y_true, segment_preds, zero_division=0)),
            "recall": float(recall_score(y_true, segment_preds, zero_division=0)),
            "f1_score": float(f1_score(y_true, segment_preds, zero_division=0)),
            "auc": float(roc_auc_score(y_true, segment_probs))
        }

        # Compute McNemar's test for comparing classifiers
        try:
            # Create contingency table
            # both correct, full correct & segment wrong, full wrong & segment correct, both wrong
            contingency_table = np.zeros((2, 2), dtype=int)

            for i in range(len(y_true)):
                full_correct = full_preds[i] == y_true[i]
                segment_correct = segment_preds[i] == y_true[i]

                if full_correct and segment_correct:
                    contingency_table[0, 0] += 1
                elif full_correct and not segment_correct:
                    contingency_table[0, 1] += 1
                elif not full_correct and segment_correct:
                    contingency_table[1, 0] += 1
                else:  # both wrong
                    contingency_table[1, 1] += 1

            # Use our custom implementation of McNemar's test
            mcnemar_test = self.compute_mcnemar_test(contingency_table)

        except Exception as e:
            print(f"Error computing McNemar's test: {e}")
            mcnemar_test = {
                "error": str(e)
            }

        # Create aggregate results dictionary
        aggregate_metrics = {
            "full_model": full_model_metrics,
            "segment_model": segment_model_metrics,
            "mcnemar_test": mcnemar_test,
            "metrics_diff": {
                metric: segment_model_metrics[metric] - full_model_metrics[metric]
                for metric in full_model_metrics
            }
        }

        return aggregate_metrics

    def compute_family_metrics(self, metrics):
        """
        Compute performance metrics grouped by malware family.

        Args:
            metrics: Dictionary containing per-sample metrics

        Returns:
            dict: Dictionary containing family-level metrics
        """
        # Only compute if family labels are available
        if 'family' not in metrics or metrics['family'] is None:
            print("Family labels not available. Skipping family-level analysis.")
            return None

        print("Computing family-level performance metrics...")

        # Extract data
        y_true = np.array(metrics['true_label'])
        full_preds = np.array(metrics['full_model']['prediction'])
        segment_preds = np.array(metrics['segment_model']['prediction'])
        full_probs = np.array(metrics['full_model']['probability'])
        segment_probs = np.array(metrics['segment_model']['probability'])
        families = metrics['family']

        # Get unique families
        unique_families = sorted(set(families))

        # Initialize results dictionary
        family_metrics = {}

        # Compute metrics for each family
        for family in unique_families:
            # Create mask for this family
            family_mask = np.array([f == family for f in families])

            # Skip if no samples for this family
            if np.sum(family_mask) == 0:
                continue

            # Extract samples for this family
            family_y_true = y_true[family_mask]
            family_full_preds = full_preds[family_mask]
            family_segment_preds = segment_preds[family_mask]
            family_full_probs = full_probs[family_mask]
            family_segment_probs = segment_probs[family_mask]

            # Calculate metrics for full model
            try:
                full_acc = accuracy_score(family_y_true, family_full_preds)
                full_prec = precision_score(family_y_true, family_full_preds, zero_division=0)
                full_rec = recall_score(family_y_true, family_full_preds, zero_division=0)
                full_f1 = f1_score(family_y_true, family_full_preds, zero_division=0)

                # Only compute AUC if we have both classes
                if len(np.unique(family_y_true)) > 1:
                    full_auc = roc_auc_score(family_y_true, family_full_probs)
                else:
                    full_auc = None
            except Exception as e:
                print(f"Error computing metrics for family {family} (full model): {e}")
                full_acc, full_prec, full_rec, full_f1, full_auc = None, None, None, None, None

            # Calculate metrics for segment model
            try:
                segment_acc = accuracy_score(family_y_true, family_segment_preds)
                segment_prec = precision_score(family_y_true, family_segment_preds, zero_division=0)
                segment_rec = recall_score(family_y_true, family_segment_preds, zero_division=0)
                segment_f1 = f1_score(family_y_true, family_segment_preds, zero_division=0)

                # Only compute AUC if we have both classes
                if len(np.unique(family_y_true)) > 1:
                    segment_auc = roc_auc_score(family_y_true, family_segment_probs)
                else:
                    segment_auc = None
            except Exception as e:
                print(f"Error computing metrics for family {family} (segment model): {e}")
                segment_acc, segment_prec, segment_rec, segment_f1, segment_auc = None, None, None, None, None

            # Try to run Wilcoxon test for this family
            try:
                # Extract metrics for this family
                family_indices = [i for i, f in enumerate(families) if f == family]
                family_full_correct = [metrics['full_model']['correct'][i] for i in family_indices]
                family_segment_correct = [metrics['segment_model']['correct'][i] for i in family_indices]

                # Run test if we have enough samples
                if len(family_indices) > 5:  # Need a minimum number for the test
                    from scipy import stats
                    wilcoxon_result = stats.wilcoxon(
                        np.array(family_full_correct).astype(int),
                        np.array(family_segment_correct).astype(int)
                    )
                    p_value = wilcoxon_result.pvalue
                    is_significant = p_value < 0.05
                else:
                    wilcoxon_result = None
                    p_value = None
                    is_significant = False
            except Exception as e:
                print(f"Error computing Wilcoxon test for family {family}: {e}")
                wilcoxon_result = None
                p_value = None
                is_significant = False

            # Store results
            family_metrics[family] = {
                "sample_count": int(np.sum(family_mask)),
                "full_model": {
                    "accuracy": float(full_acc) if full_acc is not None else None,
                    "precision": float(full_prec) if full_prec is not None else None,
                    "recall": float(full_rec) if full_rec is not None else None,
                    "f1_score": float(full_f1) if full_f1 is not None else None,
                    "auc": float(full_auc) if full_auc is not None else None
                },
                "segment_model": {
                    "accuracy": float(segment_acc) if segment_acc is not None else None,
                    "precision": float(segment_prec) if segment_prec is not None else None,
                    "recall": float(segment_rec) if segment_rec is not None else None,
                    "f1_score": float(segment_f1) if segment_f1 is not None else None,
                    "auc": float(segment_auc) if segment_auc is not None else None
                },
                "wilcoxon_test": {
                    "p_value": float(p_value) if p_value is not None else None,
                    "is_significant": bool(is_significant),
                    "better_model": "Segment Model" if segment_acc > full_acc else "Full Model" if full_acc > segment_acc else "Equal"
                }
            }

        return family_metrics

    def generate_visualizations(self, metrics, wilcoxon_results, aggregate_metrics, family_metrics=None):
        """
        Generate visualizations for the statistical comparison.

        Args:
            metrics: Dictionary containing per-sample metrics
            wilcoxon_results: Dictionary containing Wilcoxon test results
            aggregate_metrics: Dictionary containing aggregate metrics
            family_metrics: Dictionary containing family-level metrics (optional)
        """
        print("Generating visualizations...")

        # Create a subdirectory for visualizations
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Create ROC curve comparison
        plt.figure(figsize=(10, 8))

        # ROC curve for full model
        fpr_full, tpr_full, _ = roc_curve(
            metrics['true_label'],
            metrics['full_model']['probability']
        )
        roc_auc_full = auc(fpr_full, tpr_full)

        # ROC curve for segment model
        fpr_segment, tpr_segment, _ = roc_curve(
            metrics['true_label'],
            metrics['segment_model']['probability']
        )
        roc_auc_segment = auc(fpr_segment, tpr_segment)

        # Plot both curves
        plt.plot(
            fpr_full, tpr_full,
            color='blue', lw=2,
            label=f'Full Image Model (AUC = {roc_auc_full:.3f})'
        )
        plt.plot(
            fpr_segment, tpr_segment,
            color='red', lw=2,
            label=f'Segment Model (AUC = {roc_auc_segment:.3f})'
        )

        # Add diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

        # Add labels and legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # Save the figure
        plt.savefig(os.path.join(viz_dir, 'roc_curve_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Create probability distribution comparison
        plt.figure(figsize=(12, 6))

        # Use KDE for smoother distribution
        sns.kdeplot(
            metrics['full_model']['probability'],
            color='blue', fill=True, alpha=0.3,
            label='Full Image Model'
        )
        sns.kdeplot(
            metrics['segment_model']['probability'],
            color='red', fill=True, alpha=0.3,
            label='Segment Model'
        )

        # Add labels and legend
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the figure
        plt.savefig(os.path.join(viz_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Create per-sample performance comparison
        plt.figure(figsize=(12, 8))

        # Extract correct/incorrect predictions
        x = np.arange(len(metrics['true_label']))
        full_correct = np.array(metrics['full_model']['correct'])
        segment_correct = np.array(metrics['segment_model']['correct'])

        # Define color codes for different cases
        colors = []
        for full, segment in zip(full_correct, segment_correct):
            if full and segment:
                colors.append('lightgreen')  # Both correct
            elif full and not segment:
                colors.append('lightblue')  # Only full correct
            elif not full and segment:
                colors.append('salmon')  # Only segment correct
            else:
                colors.append('lightgray')  # Both wrong

        # Create the scatter plot
        plt.scatter(
            metrics['full_model']['probability'],
            metrics['segment_model']['probability'],
            c=colors, alpha=0.7, edgecolors='k', linewidths=0.5
        )

        # Add diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

        # Add labels and legend
        plt.xlabel('Full Model Probability')
        plt.ylabel('Segment Model Probability')
        plt.title('Per-Sample Prediction Comparison')

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='k', label='Both correct'),
            Patch(facecolor='lightblue', edgecolor='k', label='Only full model correct'),
            Patch(facecolor='salmon', edgecolor='k', label='Only segment model correct'),
            Patch(facecolor='lightgray', edgecolor='k', label='Both wrong')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        plt.grid(True, alpha=0.3)

        # Save the figure
        plt.savefig(os.path.join(viz_dir, 'per_sample_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Create a bar chart comparing overall metrics
        plt.figure(figsize=(12, 6))

        # Extract metrics
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        full_values = [
            aggregate_metrics['full_model']['accuracy'],
            aggregate_metrics['full_model']['precision'],
            aggregate_metrics['full_model']['recall'],
            aggregate_metrics['full_model']['f1_score'],
            aggregate_metrics['full_model']['auc']
        ]
        segment_values = [
            aggregate_metrics['segment_model']['accuracy'],
            aggregate_metrics['segment_model']['precision'],
            aggregate_metrics['segment_model']['recall'],
            aggregate_metrics['segment_model']['f1_score'],
            aggregate_metrics['segment_model']['auc']
        ]

        # Create bar positions
        x = np.arange(len(metrics_names))
        width = 0.35

        # Create bars
        plt.bar(x - width / 2, full_values, width, label='Full Image Model', color='blue', alpha=0.7)
        plt.bar(x + width / 2, segment_values, width, label='Segment Model', color='red', alpha=0.7)

        # Add labels and legend
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(full_values):
            plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center')
        for i, v in enumerate(segment_values):
            plt.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center')

        # Save the figure
        plt.savefig(os.path.join(viz_dir, 'overall_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Create family performance comparison if family metrics are available
        if family_metrics:
            # Only keep families with more than a minimum number of samples
            min_samples = 5
            valid_families = {f: m for f, m in family_metrics.items()
                              if m['sample_count'] >= min_samples and f != 'benign'}

            if valid_families:
                # Create bar chart for each metric
                for metric_name in ['accuracy', 'f1_score']:
                    plt.figure(figsize=(14, 8))

                    # Create data for plotting
                    families = list(valid_families.keys())
                    full_values = [valid_families[f]['full_model'][metric_name] for f in families]
                    segment_values = [valid_families[f]['segment_model'][metric_name] for f in families]

                    # Sort families by segment model improvement
                    improvement = [s - f for f, s in zip(full_values, segment_values)]
                    sorted_indices = np.argsort(improvement)

                    families = [families[i] for i in sorted_indices]
                    full_values = [full_values[i] for i in sorted_indices]
                    segment_values = [segment_values[i] for i in sorted_indices]

                    # Create bar positions
                    x = np.arange(len(families))
                    width = 0.35

                    # Create bars
                    plt.bar(x - width / 2, full_values, width, label='Full Image Model', color='blue', alpha=0.7)
                    plt.bar(x + width / 2, segment_values, width, label='Segment Model', color='red', alpha=0.7)

                    # Add sample count for each family
                    for i, family in enumerate(families):
                        count = valid_families[family]['sample_count']
                        plt.text(i, -0.05, f'n={count}', ha='center', fontsize=8)

                    # Add significance markers
                    for i, family in enumerate(families):
                        if valid_families[family]['wilcoxon_test']['is_significant']:
                            plt.text(i, max(full_values[i], segment_values[i]) + 0.02, '*',
                                     ha='center', fontsize=16, color='black')

                    # Add labels and legend
                    plt.xlabel('Malware Family')
                    plt.ylabel(f'{metric_name.replace("_", " ").title()}')
                    plt.title(f'Per-Family {metric_name.replace("_", " ").title()} Comparison')
                    plt.xticks(x, families, rotation=45, ha='right')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # Save the figure
                    plt.savefig(os.path.join(viz_dir, f'family_{metric_name}_comparison.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()

        # 5. Create family performance comparison if family metrics are available
        if family_metrics:
            # Only keep families with more than a minimum number of samples
            min_samples = 5
            valid_families = {f: m for f, m in family_metrics.items()
                              if m['sample_count'] >= min_samples and f != 'benign'}

            if valid_families:
                # Create bar chart for each metric
                for metric_name in ['accuracy', 'f1_score']:
                    plt.figure(figsize=(14, 8))

                    # Create data for plotting
                    families = list(valid_families.keys())
                    full_values = [valid_families[f]['full_model'][metric_name] for f in families]
                    segment_values = [valid_families[f]['segment_model'][metric_name] for f in families]

                    # Sort families by segment model improvement
                    improvement = [s - f for f, s in zip(full_values, segment_values)]
                    sorted_indices = np.argsort(improvement)

                    families = [families[i] for i in sorted_indices]
                    full_values = [full_values[i] for i in sorted_indices]
                    segment_values = [segment_values[i] for i in sorted_indices]

                    # Create bar positions
                    x = np.arange(len(families))
                    width = 0.35

                    # Create bars
                    plt.bar(x - width / 2, full_values, width, label='Full Image Model', color='blue', alpha=0.7)
                    plt.bar(x + width / 2, segment_values, width, label='Segment Model', color='red', alpha=0.7)

                    # Add sample count for each family
                    for i, family in enumerate(families):
                        count = valid_families[family]['sample_count']
                        plt.text(i, -0.05, f'n={count}', ha='center', fontsize=8)

                    # Add significance markers
                    for i, family in enumerate(families):
                        if valid_families[family]['wilcoxon_test']['is_significant']:
                            plt.text(i, max(full_values[i], segment_values[i]) + 0.02, '*',
                                     ha='center', fontsize=16, color='black')

                    # Add labels and legend
                    plt.xlabel('Malware Family')
                    plt.ylabel(f'{metric_name.replace("_", " ").title()}')
                    plt.title(f'Per-Family {metric_name.replace("_", " ").title()} Comparison')
                    plt.xticks(x, families, rotation=45, ha='right')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # Save the figure
                    plt.savefig(os.path.join(viz_dir, f'family_{metric_name}_comparison.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()

    def save_results(self, metrics, wilcoxon_results, aggregate_metrics, family_metrics=None):
        """
        Save all results to JSON files.

        Args:
            metrics: Dictionary containing per-sample metrics
            wilcoxon_results: Dictionary containing Wilcoxon test results
            aggregate_metrics: Dictionary containing aggregate metrics
            family_metrics: Dictionary containing family-level metrics (optional)
        """
        print("Saving results to files...")

        # Save Wilcoxon test results
        with open(os.path.join(self.output_dir, 'wilcoxon_test_results.json'), 'w') as f:
            json.dump(wilcoxon_results, f, indent=4)

        # Save aggregate metrics
        with open(os.path.join(self.output_dir, 'aggregate_metrics.json'), 'w') as f:
            json.dump(aggregate_metrics, f, indent=4)

        # Save family metrics if available
        if family_metrics:
            with open(os.path.join(self.output_dir, 'family_metrics.json'), 'w') as f:
                json.dump(family_metrics, f, indent=4)

        # Save per-sample metrics as a CSV file for further analysis
        try:
            # Convert the nested metrics dict to a flat DataFrame
            df_data = []
            for i in range(len(metrics['sample_idx'])):
                row = {
                    'sample_idx': metrics['sample_idx'][i],
                    'file_path': metrics['file_path'][i],
                    'true_label': metrics['true_label'][i],
                    'full_prediction': metrics['full_model']['prediction'][i],
                    'full_probability': metrics['full_model']['probability'][i],
                    'full_correct': metrics['full_model']['correct'][i],
                    'full_log_loss': metrics['full_model']['log_loss'][i],
                    'segment_prediction': metrics['segment_model']['prediction'][i],
                    'segment_probability': metrics['segment_model']['probability'][i],
                    'segment_correct': metrics['segment_model']['correct'][i],
                    'segment_log_loss': metrics['segment_model']['log_loss'][i],
                    'prediction_match': metrics['performance_diff']['prediction_match'][i],
                    'probability_diff': metrics['performance_diff']['probability_diff'][i],
                    'log_loss_diff': metrics['performance_diff']['log_loss_diff'][i]
                }

                # Add family if available
                if metrics['family'] is not None:
                    row['family'] = metrics['family'][i]

                df_data.append(row)

            # Create DataFrame and save to CSV
            df = pd.DataFrame(df_data)
            df.to_csv(os.path.join(self.output_dir, 'per_sample_metrics.csv'), index=False)
        except Exception as e:
            print(f"Error saving per-sample metrics to CSV: {e}")

    def create_summary_report(self, wilcoxon_results, aggregate_metrics, family_metrics=None):
        """
        Create a summary report in Markdown format.

        Args:
            wilcoxon_results: Dictionary containing Wilcoxon test results
            aggregate_metrics: Dictionary containing aggregate metrics
            family_metrics: Dictionary containing family-level metrics (optional)

        Returns:
            str: Markdown formatted summary report
        """
        print("Creating summary report...")

        # Initialize report
        report = []
        report.append("# Statistical Comparison Report: Full Image vs. Segment Models\n")

        # Add date
        from datetime import datetime
        report.append(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Overall results section
        report.append("## Overall Results\n")

        # Wilcoxon signed-rank test results
        report.append("### Wilcoxon Signed-Rank Test Results\n")

        # Accuracy test
        acc_test = wilcoxon_results["wilcoxon_test_accuracy"]
        p_value_acc = acc_test.get("p_value")
        acc_result = "statistically significant" if acc_test["is_significant"] else "not statistically significant"
        acc_better = acc_test["better_model"]

        report.append(f"**Classification Accuracy Test:**\n")
        report.append(f"- Result: {acc_result}")
        if p_value_acc is not None:  # This is the variable from above
            if p_value_acc < 0.0001:
                report.append(f" (p-value: {p_value_acc:.2e})\n")
            else:
                report.append(f" (p-value: {p_value_acc:.4f})\n")
        else:
            report.append(" (p-value: N/A)\n")
        report.append(f"- Better model: {acc_better}\n")
        report.append(f"- Full model accuracy: {acc_test['full_model_accuracy']:.4f}\n")
        report.append(f"- Segment model accuracy: {acc_test['segment_model_accuracy']:.4f}\n")
        report.append(f"- Absolute difference: {abs(acc_test['accuracy_difference']):.4f}\n")

        # Log loss test
        loss_test = wilcoxon_results["wilcoxon_test_log_loss"]
        loss_result = "statistically significant" if loss_test["is_significant"] else "not statistically significant"
        loss_better = loss_test["better_model"]

        report.append(f"\n**Log Loss Test:**\n")
        report.append(f"- Result: {loss_result} (p-value: {loss_test['p_value']:.4f})\n")
        report.append(f"- Better model: {loss_better}\n")
        report.append(f"- Full model log loss: {loss_test['full_model_log_loss']:.4f}\n")
        report.append(f"- Segment model log loss: {loss_test['segment_model_log_loss']:.4f}\n")
        report.append(f"- Absolute difference: {abs(loss_test['log_loss_difference']):.4f}\n")

        # McNemar's test
        mcnemar = aggregate_metrics["mcnemar_test"]
        if "error" not in mcnemar:
            contingency = mcnemar["contingency_table"]
            mcnemar_result = "statistically significant" if mcnemar[
                "is_significant"] else "not statistically significant"
            mcnemar_better = mcnemar["better_model"]

            report.append(f"\n### McNemar's Test Results\n")
            report.append(f"- Result: {mcnemar_result} (p-value: {mcnemar['p_value']:.4f})\n")
            report.append(f"- Better model: {mcnemar_better}\n")
            report.append(f"- Contingency table:\n")
            report.append(f"  * Both models correct: {contingency[0][0]}\n")
            report.append(f"  * Only full model correct: {contingency[0][1]}\n")
            report.append(f"  * Only segment model correct: {contingency[1][0]}\n")
            report.append(f"  * Both models incorrect: {contingency[1][1]}\n")

        # Performance metrics comparison
        report.append(f"\n### Performance Metrics Comparison\n")
        report.append(f"| Metric | Full Model | Segment Model | Difference | % Improvement |\n")
        report.append(f"|--------|------------|---------------|------------|---------------|\n")

        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            full_val = aggregate_metrics["full_model"][metric]
            seg_val = aggregate_metrics["segment_model"][metric]
            diff = aggregate_metrics["metrics_diff"][metric]
            pct_change = (diff / full_val) * 100 if full_val != 0 else float('inf')

            # Format the sign of the difference and percentage
            diff_sign = "+" if diff > 0 else ""
            pct_sign = "+" if diff > 0 else ""

            report.append(
                f"| {metric.replace('_', ' ').title()} | {full_val:.4f} | {seg_val:.4f} | {diff_sign}{diff:.4f} | {pct_sign}{pct_change:.2f}% |\n")

        # Family-level analysis if available
        if family_metrics:
            report.append(f"\n## Family-Level Analysis\n")

            # Get families with sufficient samples
            valid_families = {f: m for f, m in family_metrics.items()
                              if m['sample_count'] >= 5 and f != 'benign'}

            if valid_families:
                # Sort families by improvement in F1 score
                def get_f1_improvement(family_data):
                    full_f1 = family_data['full_model']['f1_score'] or 0
                    seg_f1 = family_data['segment_model']['f1_score'] or 0
                    return seg_f1 - full_f1

                sorted_families = sorted(
                    valid_families.items(),
                    key=lambda x: get_f1_improvement(x[1]),
                    reverse=True
                )

                # Create a table of results
                report.append(f"### Per-Family Performance Comparison\n")
                report.append(f"| Family | Count | Full Model F1 | Segment Model F1 | Difference | Significant? |\n")
                report.append(f"|--------|-------|---------------|------------------|------------|-------------|\n")

                for family, data in sorted_families:
                    count = data['sample_count']
                    full_f1 = data['full_model']['f1_score'] or 0
                    seg_f1 = data['segment_model']['f1_score'] or 0
                    diff = seg_f1 - full_f1
                    sig = "Yes" if data['wilcoxon_test']['is_significant'] else "No"

                    # Format the sign of the difference
                    diff_sign = "+" if diff > 0 else ""

                    report.append(
                        f"| {family} | {count} | {full_f1:.4f} | {seg_f1:.4f} | {diff_sign}{diff:.4f} | {sig} |\n")

                # Highlight key findings
                report.append(f"\n### Key Family-Level Findings\n")

                # Most improved families
                improved_families = [f for f, d in sorted_families if get_f1_improvement(d) > 0]
                if improved_families:
                    top_improved = improved_families[:3] if len(improved_families) > 3 else improved_families
                    report.append(f"- **Most improved families with segment model:** {', '.join(top_improved)}\n")

                # Degraded families
                degraded_families = [f for f, d in sorted_families if get_f1_improvement(d) < 0]
                if degraded_families:
                    most_degraded = degraded_families[-3:] if len(degraded_families) > 3 else degraded_families
                    report.append(
                        f"- **Families that performed worse with segment model:** {', '.join(most_degraded)}\n")

                # Families with significant differences
                sig_families = [f for f, d in sorted_families if d['wilcoxon_test']['is_significant']]
                if sig_families:
                    report.append(
                        f"- **Families with statistically significant differences:** {', '.join(sig_families)}\n")

        # Conclusions
        report.append(f"\n## Conclusions\n")

        # Determine overall winner
        if wilcoxon_results["wilcoxon_test_accuracy"]["is_significant"]:
            winner = wilcoxon_results["wilcoxon_test_accuracy"]["better_model"]
            report.append(
                f"- The {winner} shows **statistically significant better performance** in terms of classification accuracy.\n")
        else:
            report.append(
                f"- There is **no statistically significant difference** in classification accuracy between the two models.\n")

        if wilcoxon_results["wilcoxon_test_log_loss"]["is_significant"]:
            winner = wilcoxon_results["wilcoxon_test_log_loss"]["better_model"]
            report.append(
                f"- The {winner} shows **statistically significant better performance** in terms of log loss (prediction confidence).\n")
        else:
            report.append(
                f"- There is **no statistically significant difference** in log loss (prediction confidence) between the two models.\n")

        # McNemar's test conclusion
        if "error" not in mcnemar:
            if mcnemar["is_significant"]:
                report.append(
                    f"- McNemar's test shows a **statistically significant difference** in the pattern of errors between the two models, with the {mcnemar['better_model']} performing better.\n")
            else:
                report.append(
                    f"- McNemar's test shows **no statistically significant difference** in the pattern of errors between the two models.\n")

        # Overall recommendation
        if wilcoxon_results["wilcoxon_test_accuracy"]["is_significant"] or wilcoxon_results["wilcoxon_test_log_loss"][
            "is_significant"]:
            # Determine which model is better overall
            acc_better = wilcoxon_results["wilcoxon_test_accuracy"]["better_model"]
            loss_better = wilcoxon_results["wilcoxon_test_log_loss"]["better_model"]

            if acc_better == loss_better:
                report.append(
                    f"\n**Overall recommendation:** The {acc_better} is recommended based on consistent superior performance.\n")
            else:
                report.append(
                    f"\n**Overall recommendation:** Mixed results - the {acc_better} has better accuracy, while the {loss_better} has better prediction confidence.\n")
        else:
            # Check which has better average metrics
            full_avg = sum(aggregate_metrics["full_model"].values()) / len(aggregate_metrics["full_model"])
            seg_avg = sum(aggregate_metrics["segment_model"].values()) / len(aggregate_metrics["segment_model"])
            better = "Segment Model" if seg_avg > full_avg else "Full Model"

            report.append(
                f"\n**Overall recommendation:** No clear statistical winner. The {better} has slightly better average metrics, but differences are not statistically significant.\n")

        # Join the report into a single string
        return "\n".join(report)

    def run_analysis(self):
        """
        Run the complete statistical comparison analysis.

        Returns:
            dict: Dictionary containing all results
        """
        # Load data
        if not self.load_test_data():
            print("Failed to load test data. Aborting analysis.")
            return None

        # Calculate per-sample metrics
        metrics = self.calculate_paired_metrics()

        # Run Wilcoxon test
        wilcoxon_results = self.run_wilcoxon_test(metrics)

        # Compute aggregate metrics
        aggregate_metrics = self.compute_aggregate_metrics(metrics)

        # Compute family-level metrics if available
        family_metrics = self.compute_family_metrics(metrics) if self.family_labels is not None else None

        # Generate visualizations
        self.generate_visualizations(metrics, wilcoxon_results, aggregate_metrics, family_metrics)

        # Save results
        self.save_results(metrics, wilcoxon_results, aggregate_metrics, family_metrics)

        # Create summary report
        report = self.create_summary_report(wilcoxon_results, aggregate_metrics, family_metrics)

        # Save report
        report_path = os.path.join(self.output_dir, 'summary_report.md')
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Analysis complete. Summary report saved to {report_path}")

        # Return results
        return {
            'wilcoxon_results': wilcoxon_results,
            'aggregate_metrics': aggregate_metrics,
            'family_metrics': family_metrics,
            'report': report
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Statistical Comparison of Malware Classification Models')

    parser.add_argument('--full-model-dir', type=str, required=True,
                        help='Directory containing full image model results')
    parser.add_argument('--segment-model-dir', type=str, required=True,
                        help='Directory containing segment model results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save comparison results')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Set numpy random seed for reproducibility
    np.random.seed(42)

    # Initialize and run the statistical comparison
    comparator = StatisticalComparison(
        full_model_dir=args.full_model_dir,
        segment_model_dir=args.segment_model_dir,
        output_dir=args.output_dir
    )

    # Run the analysis
    results = comparator.run_analysis()

    # Print key results to console
    if results:
        wilcoxon_acc = results['wilcoxon_results']['wilcoxon_test_accuracy']
        wilcoxon_loss = results['wilcoxon_results']['wilcoxon_test_log_loss']

        print("\n--- Key Results ---")

        # Handle possible None values for p-values
        acc_p = wilcoxon_acc.get('p_value')
        if acc_p is not None:
            # Use scientific notation for very small p-values
            if acc_p < 0.0001:
                print(f"Wilcoxon test (accuracy): p-value={acc_p:.2e}, "
                      f"significant={wilcoxon_acc['is_significant']}, "
                      f"better model={wilcoxon_acc['better_model']}")
            else:
                print(f"Wilcoxon test (accuracy): p-value={acc_p:.4f}, "
                      f"significant={wilcoxon_acc['is_significant']}, "
                      f"better model={wilcoxon_acc['better_model']}")
        else:
            print(f"Wilcoxon test (accuracy): p-value=N/A, "
                  f"significant={wilcoxon_acc['is_significant']}, "
                  f"better model={wilcoxon_acc['better_model']}")

        # Print some aggregate metrics
        full_acc = results['aggregate_metrics']['full_model']['accuracy']
        seg_acc = results['aggregate_metrics']['segment_model']['accuracy']
        full_f1 = results['aggregate_metrics']['full_model']['f1_score']
        seg_f1 = results['aggregate_metrics']['segment_model']['f1_score']

        print(f"Full model: accuracy={full_acc:.4f}, F1-score={full_f1:.4f}")
        print(f"Segment model: accuracy={seg_acc:.4f}, F1-score={seg_f1:.4f}")

        print(f"\nFull results saved to {comparator.output_dir}")

    return 0


if __name__ == "__main__":
    main()
