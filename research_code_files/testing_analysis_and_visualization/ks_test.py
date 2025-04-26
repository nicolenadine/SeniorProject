#!/usr/bin/env python3
"""
Statistical Analysis Module for Malware Classification Project

This module performs statistical tests to compare performance distributions
between the full-image and segmented model approaches, with a focus on
the Kolmogorov-Smirnov test for distribution comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import tensorflow as tf
from matplotlib.ticker import PercentFormatter
import warnings

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Class for performing statistical analysis between full-image and segmented models.
    """

    def __init__(self, full_model_dir, segmented_model_dir, output_dir=None):
        """
        Initialize the statistical analyzer.

        Args:
            full_model_dir (str): Directory containing full-image model results
            segmented_model_dir (str): Directory containing segmented model results
            output_dir (str): Directory to save analysis results
        """
        self.full_model_dir = full_model_dir
        self.segmented_model_dir = segmented_model_dir

        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(full_model_dir), 'statistical_analysis')
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Create subdirectories
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.tables_dir = os.path.join(self.output_dir, 'tables')
        self.family_dir = os.path.join(self.output_dir, 'family_analysis')

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.family_dir, exist_ok=True)

        # Data to be loaded
        self.full_probabilities = None
        self.segmented_probabilities = None
        self.test_files = None
        self.test_labels = None
        self.family_labels = None
        self.file_family_map = {}

        # Results storage
        self.ks_results = {}
        self.per_family_results = {}
        self.performance_metrics = {}

    def load_data(self):
        """
        Load prediction data from both models.

        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        print("Loading data from both models...")

        # Load full model predictions
        try:
            self.full_probabilities = np.load(os.path.join(self.full_model_dir, 'metrics', 'y_pred_prob.npy'))
            if len(self.full_probabilities.shape) > 1 and self.full_probabilities.shape[1] == 1:
                self.full_probabilities = self.full_probabilities.flatten()

            # Try alternate locations if not found
            if self.full_probabilities is None:
                self.full_probabilities = np.load(os.path.join(self.full_model_dir, 'test_probabilities.npy'))
        except Exception as e:
            print(f"Error loading full model probabilities: {e}")
            # Try alternate locations
            try:
                self.full_probabilities = np.load(os.path.join(self.full_model_dir, 'test_probabilities.npy'))
            except:
                print(f"Could not find full model probabilities in {self.full_model_dir}")
                return False

        # Load segmented model predictions
        try:
            self.segmented_probabilities = np.load(os.path.join(self.segmented_model_dir, 'test_probabilities.npy'))
            if len(self.segmented_probabilities.shape) > 1 and self.segmented_probabilities.shape[1] == 1:
                self.segmented_probabilities = self.segmented_probabilities.flatten()
        except Exception as e:
            print(f"Error loading segmented model probabilities: {e}")
            return False

        # Load test file paths and labels
        try:
            self.test_files = np.load(os.path.join(self.full_model_dir, 'metrics', 'test_file_paths.npy'))
            self.test_labels = np.load(os.path.join(self.full_model_dir, 'metrics', 'y_true.npy'))

            # If not found in metrics directory, try data_splits directory
            if self.test_files is None or self.test_labels is None:
                splits_dir = os.path.join(self.full_model_dir, 'data_splits')
                with open(os.path.join(splits_dir, 'test_files.txt'), 'r') as f:
                    self.test_files = np.array(f.read().splitlines())
                with open(os.path.join(splits_dir, 'test_labels.txt'), 'r') as f:
                    self.test_labels = np.array([int(x) for x in f.read().splitlines()])
        except Exception as e:
            print(f"Warning: Could not load test paths and labels: {e}")
            print("Will continue without file path information")

        # Load family information if available
        try:
            # Try loading from segment selection metadata first
            metadata_path = os.path.join(self.segmented_model_dir, 'segment_selection_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    test_metadata = metadata.get('test', [])
                    self.family_labels = [item.get('family', 'Unknown') for item in test_metadata]

                    # Create file to family mapping
                    self.file_family_map = {item.get('file_path', ''): item.get('family', 'Unknown')
                                            for item in test_metadata}
            else:
                # Try another approach - check for family_labels.txt
                splits_dir = os.path.join(self.full_model_dir, 'data_splits')
                family_path = os.path.join(splits_dir, 'family_labels.txt')
                if os.path.exists(family_path):
                    with open(family_path, 'r') as f:
                        all_family_labels = f.read().splitlines()

                    # If we have test file indices, we can extract test family labels
                    if self.test_files is not None:
                        # We need to match the family labels to test files
                        # This depends on how your DataHandler class stores indices
                        print("Family information available, but need to match with test indices.")
                        self.family_labels = ['Unknown'] * len(self.test_files)
        except Exception as e:
            print(f"Warning: Could not load family information: {e}")
            print("Will continue without family-level analysis")

        # Sanity check on data lengths
        if len(self.full_probabilities) != len(self.segmented_probabilities):
            print(f"Warning: Probability array lengths don't match: "
                  f"full={len(self.full_probabilities)}, segmented={len(self.segmented_probabilities)}")
            return False

        print(f"Data loaded successfully. Test set size: {len(self.full_probabilities)} samples")
        return True

    def perform_ks_test(self):
        """
        Perform Kolmogorov-Smirnov test to compare probability distributions.

        Returns:
            dict: Dictionary containing KS test results
        """
        print("Performing Kolmogorov-Smirnov test on probability distributions...")

        # Ensure data is loaded
        if self.full_probabilities is None or self.segmented_probabilities is None:
            if not self.load_data():
                print("Failed to load data. Cannot perform KS test.")
                return {}

        # Perform KS test on full dataset
        ks_statistic, p_value = stats.ks_2samp(self.full_probabilities, self.segmented_probabilities)

        self.ks_results = {
            'statistic': float(ks_statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),  # Convert numpy.bool_ to Python bool
            'sample_size': int(len(self.full_probabilities))
        }

        # Separate KS tests for benign and malware samples, if labels are available
        if self.test_labels is not None:
            benign_indices = np.where(self.test_labels == 0)[0]
            malware_indices = np.where(self.test_labels == 1)[0]

            # KS test for benign samples
            benign_ks, benign_p = stats.ks_2samp(
                self.full_probabilities[benign_indices],
                self.segmented_probabilities[benign_indices]
            )

            # KS test for malware samples
            malware_ks, malware_p = stats.ks_2samp(
                self.full_probabilities[malware_indices],
                self.segmented_probabilities[malware_indices]
            )

            self.ks_results['benign'] = {
                'statistic': float(benign_ks),
                'p_value': float(benign_p),
                'significant': bool(benign_p < 0.05),  # Convert numpy.bool_ to Python bool
                'sample_size': int(len(benign_indices))
            }

            self.ks_results['malware'] = {
                'statistic': float(malware_ks),
                'p_value': float(malware_p),
                'significant': bool(malware_p < 0.05),  # Convert numpy.bool_ to Python bool
                'sample_size': int(len(malware_indices))
            }

        # Print results
        print(f"KS test results (overall): statistic={ks_statistic:.4f}, p-value={p_value:.4g}")
        if p_value < 0.05:
            print("The distributions are significantly different (p < 0.05)")
        else:
            print("No significant difference detected between distributions (p >= 0.05)")

        if 'benign' in self.ks_results:
            print(f"KS test results (benign): statistic={benign_ks:.4f}, p-value={benign_p:.4g}")
            print(f"KS test results (malware): statistic={malware_ks:.4f}, p-value={malware_p:.4g}")

        # Save results to JSON
        with open(os.path.join(self.output_dir, 'ks_test_results.json'), 'w') as f:
            json.dump(self.ks_results, f, indent=4)

        return self.ks_results

    def family_level_analysis(self):
        """
        Perform family-level analysis if family information is available.

        Returns:
            dict: Dictionary containing family-level KS test results
        """
        if self.family_labels is None and self.test_files is None:
            print("Family labels or test files not available. Cannot perform family-level analysis.")
            return {}

        print("Performing family-level statistical analysis...")

        # If we need to infer family from file paths
        if self.family_labels is None and self.test_files is not None:
            # Extract family from file path (assuming path contains family name)
            self.family_labels = []
            for file_path in self.test_files:
                parts = str(file_path).split(os.sep)
                if 'malware' in parts:
                    try:
                        idx = parts.index('malware')
                        if idx + 1 < len(parts):
                            self.family_labels.append(parts[idx + 1])
                        else:
                            self.family_labels.append('Unknown')
                    except ValueError:
                        self.family_labels.append('Unknown')
                else:
                    self.family_labels.append('Benign')

        # Convert to numpy array
        self.family_labels = np.array(self.family_labels)

        # Analyze per family
        unique_families = np.unique(self.family_labels)
        print(f"Found {len(unique_families)} unique families/classes")

        family_results = {}
        for family in unique_families:
            if family == 'Unknown':
                continue

            # Get indices for this family
            family_indices = np.where(self.family_labels == family)[0]

            if len(family_indices) < 10:  # Skip families with too few samples
                print(f"Skipping {family} - only {len(family_indices)} samples")
                continue

            # Perform KS test for this family
            family_ks, family_p = stats.ks_2samp(
                self.full_probabilities[family_indices],
                self.segmented_probabilities[family_indices]
            )

            # Calculate performance metrics for this family
            full_accuracy = np.mean((self.full_probabilities[family_indices] > 0.5).astype(int) ==
                                    self.test_labels[family_indices])
            segmented_accuracy = np.mean((self.segmented_probabilities[family_indices] > 0.5).astype(int) ==
                                         self.test_labels[family_indices])

            family_results[family] = {
                'ks_statistic': float(family_ks),
                'p_value': float(family_p),
                'significant': bool(family_p < 0.05),  # Convert numpy.bool_ to Python bool
                'sample_size': int(len(family_indices)),
                'full_accuracy': float(full_accuracy),
                'segmented_accuracy': float(segmented_accuracy),
                'accuracy_diff': float(segmented_accuracy - full_accuracy)
            }

            print(f"Family: {family} (n={len(family_indices)})")
            print(f"  KS statistic: {family_ks:.4f}, p-value: {family_p:.4g}")
            print(f"  Accuracy: Full={full_accuracy:.4f}, Segmented={segmented_accuracy:.4f}, "
                  f"Diff={segmented_accuracy - full_accuracy:.4f}")

        # Save family results
        self.per_family_results = family_results
        with open(os.path.join(self.family_dir, 'family_level_results.json'), 'w') as f:
            json.dump(family_results, f, indent=4)

        # Create visualizations of family-level differences
        self.visualize_family_differences()

        return family_results

    def visualize_family_differences(self):
        """
        Create visualizations to compare family-level performance differences.
        """
        if not self.per_family_results:
            return

        # Prepare data for plots
        families = list(self.per_family_results.keys())
        sample_sizes = [self.per_family_results[f]['sample_size'] for f in families]
        full_acc = [self.per_family_results[f]['full_accuracy'] for f in families]
        segmented_acc = [self.per_family_results[f]['segmented_accuracy'] for f in families]
        acc_diff = [self.per_family_results[f]['accuracy_diff'] for f in families]

        # Sort by sample size for better visualization
        sorted_indices = np.argsort(sample_sizes)[::-1]  # Descending order
        families = [families[i] for i in sorted_indices]
        sample_sizes = [sample_sizes[i] for i in sorted_indices]
        full_acc = [full_acc[i] for i in sorted_indices]
        segmented_acc = [segmented_acc[i] for i in sorted_indices]
        acc_diff = [acc_diff[i] for i in sorted_indices]

        # Create comparison bar chart
        plt.figure(figsize=(12, 8))
        bar_width = 0.35
        indices = np.arange(len(families))

        plt.bar(indices - bar_width / 2, full_acc, bar_width, label='Full Image Model')
        plt.bar(indices + bar_width / 2, segmented_acc, bar_width, label='Segmented Model')

        plt.xlabel('Malware Family')
        plt.ylabel('Accuracy')
        plt.title('Per-Family Accuracy Comparison')
        plt.xticks(indices, families, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(self.family_dir, 'family_accuracy_comparison.png'), dpi=300)
        plt.close()

        # Create accuracy difference chart
        plt.figure(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in acc_diff]
        plt.bar(families, acc_diff, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Malware Family')
        plt.ylabel('Accuracy Difference (Segmented - Full)')
        plt.title('Performance Impact of Segmentation by Family')
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(acc_diff):
            plt.text(i, v + 0.01 if v > 0 else v - 0.03, f'{v:.3f}',
                     ha='center', va='center', fontsize=9, rotation=90)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(self.family_dir, 'family_accuracy_difference.png'), dpi=300)
        plt.close()

        # Create scatter plot of sample size vs. accuracy difference
        plt.figure(figsize=(10, 8))
        plt.scatter(sample_sizes, acc_diff, s=100, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Add family labels to points
        for i, family in enumerate(families):
            plt.annotate(family, (sample_sizes[i], acc_diff[i]),
                         xytext=(5, 5), textcoords='offset points')

        plt.xscale('log')  # Log scale for sample size
        plt.xlabel('Sample Size (log scale)')
        plt.ylabel('Accuracy Difference (Segmented - Full)')
        plt.title('Relationship Between Sample Size and Performance Difference')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.family_dir, 'sample_size_vs_difference.png'), dpi=300)
        plt.close()

    def compare_probability_distributions(self):
        """
        Visualize and compare probability distributions between the two models.
        """
        print("Creating probability distribution visualizations...")

        # Ensure data is loaded
        if self.full_probabilities is None or self.segmented_probabilities is None:
            if not self.load_data():
                print("Failed to load data. Cannot compare distributions.")
                return

        # Create histogram with KDE overlay
        plt.figure(figsize=(12, 8))

        # Histogram of both distributions
        sns.histplot(self.full_probabilities, bins=50, alpha=0.5, label='Full Image Model',
                     kde=True, stat='density', color='blue')
        sns.histplot(self.segmented_probabilities, bins=50, alpha=0.5, label='Segmented Model',
                     kde=True, stat='density', color='red')

        plt.xlabel('Predicted Probability (Malware)')
        plt.ylabel('Density')
        plt.title('Distribution Comparison of Model Prediction Probabilities')
        plt.legend()
        plt.grid(alpha=0.3)

        # Add KS statistic and p-value to plot if available
        if self.ks_results:
            plt.text(0.05, 0.95,
                     f"KS statistic: {self.ks_results['statistic']:.4f}\n"
                     f"p-value: {self.ks_results['p_value']:.4g}\n"
                     f"Significant: {self.ks_results['significant']}",
                     transform=plt.gca().transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'probability_distribution_comparison.png'), dpi=300)
        plt.close()

        # Create separate distributions for benign and malware samples if available
        if self.test_labels is not None:
            benign_indices = np.where(self.test_labels == 0)[0]
            malware_indices = np.where(self.test_labels == 1)[0]

            # For benign samples
            plt.figure(figsize=(12, 8))
            sns.histplot(self.full_probabilities[benign_indices], bins=50, alpha=0.5,
                         label='Full Image Model', kde=True, stat='density', color='blue')
            sns.histplot(self.segmented_probabilities[benign_indices], bins=50, alpha=0.5,
                         label='Segmented Model', kde=True, stat='density', color='red')

            plt.xlabel('Predicted Probability (Malware)')
            plt.ylabel('Density')
            plt.title('Distribution Comparison for Benign Samples')
            plt.legend()
            plt.grid(alpha=0.3)

            # Add KS statistic and p-value if available
            if 'benign' in self.ks_results:
                plt.text(0.05, 0.95,
                         f"KS statistic: {self.ks_results['benign']['statistic']:.4f}\n"
                         f"p-value: {self.ks_results['benign']['p_value']:.4g}\n"
                         f"Significant: {self.ks_results['benign']['significant']}",
                         transform=plt.gca().transAxes, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'benign_probability_distribution.png'), dpi=300)
            plt.close()

            # For malware samples
            plt.figure(figsize=(12, 8))
            sns.histplot(self.full_probabilities[malware_indices], bins=50, alpha=0.5,
                         label='Full Image Model', kde=True, stat='density', color='blue')
            sns.histplot(self.segmented_probabilities[malware_indices], bins=50, alpha=0.5,
                         label='Segmented Model', kde=True, stat='density', color='red')

            plt.xlabel('Predicted Probability (Malware)')
            plt.ylabel('Density')
            plt.title('Distribution Comparison for Malware Samples')
            plt.legend()
            plt.grid(alpha=0.3)

            # Add KS statistic and p-value if available
            if 'malware' in self.ks_results:
                plt.text(0.05, 0.95,
                         f"KS statistic: {self.ks_results['malware']['statistic']:.4f}\n"
                         f"p-value: {self.ks_results['malware']['p_value']:.4g}\n"
                         f"Significant: {self.ks_results['malware']['significant']}",
                         transform=plt.gca().transAxes, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'malware_probability_distribution.png'), dpi=300)
            plt.close()

        # Create ECDF plot (Empirical Cumulative Distribution Function)
        plt.figure(figsize=(12, 8))

        # Sort the data and calculate the ECDF
        sorted_full = np.sort(self.full_probabilities)
        sorted_segmented = np.sort(self.segmented_probabilities)
        ecdf_full = np.arange(1, len(sorted_full) + 1) / len(sorted_full)
        ecdf_segmented = np.arange(1, len(sorted_segmented) + 1) / len(sorted_segmented)

        plt.plot(sorted_full, ecdf_full, label='Full Image Model', linewidth=2)
        plt.plot(sorted_segmented, ecdf_segmented, label='Segmented Model', linewidth=2)

        plt.xlabel('Predicted Probability (Malware)')
        plt.ylabel('ECDF')
        plt.title('Empirical Cumulative Distribution Function Comparison')
        plt.legend()
        plt.grid(alpha=0.3)

        # Highlight the maximum difference (KS statistic) if available
        if self.ks_results:
            ks_stat = self.ks_results['statistic']
            # Find the point of maximum difference
            diffs = np.abs(ecdf_full - ecdf_segmented)
            max_diff_idx = np.argmax(diffs)
            max_diff_x = sorted_full[max_diff_idx]
            max_diff_y1 = ecdf_full[max_diff_idx]
            max_diff_y2 = ecdf_segmented[np.abs(sorted_segmented - max_diff_x).argmin()]

            plt.plot([max_diff_x, max_diff_x], [max_diff_y1, max_diff_y2], 'r--', linewidth=1.5)
            plt.plot(max_diff_x, max_diff_y1, 'ro')
            plt.plot(max_diff_x, max_diff_y2, 'ro')

            plt.annotate(f'KS statistic = {ks_stat:.4f}',
                         xy=(max_diff_x, (max_diff_y1 + max_diff_y2) / 2),
                         xytext=(max_diff_x + 0.1, (max_diff_y1 + max_diff_y2) / 2),
                         arrowprops=dict(arrowstyle='->'))

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'ecdf_comparison.png'), dpi=300)
        plt.close()

        # Create QQ plot
        plt.figure(figsize=(10, 10))

        # Get quantiles from both distributions
        quantiles = np.linspace(0, 1, 100)
        full_quantiles = np.quantile(self.full_probabilities, quantiles)
        segmented_quantiles = np.quantile(self.segmented_probabilities, quantiles)

        plt.plot(full_quantiles, segmented_quantiles, 'o', markersize=4)

        # Add reference line
        min_val = min(np.min(full_quantiles), np.min(segmented_quantiles))
        max_val = max(np.max(full_quantiles), np.max(segmented_quantiles))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('Full Image Model Quantiles')
        plt.ylabel('Segmented Model Quantiles')
        plt.title('QQ Plot: Segmented vs. Full Image Model Probabilities')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'qq_plot.png'), dpi=300)
        plt.close()

    def compare_performance_metrics(self):
        """
        Compare performance metrics between full image and segmented models.
        """
        print("Comparing performance metrics between models...")

        # Ensure data is loaded
        if self.full_probabilities is None or self.segmented_probabilities is None or self.test_labels is None:
            if not self.load_data():
                print("Failed to load data. Cannot compare performance metrics.")
                return {}

        # Calculate basic metrics
        full_predictions = (self.full_probabilities > 0.5).astype(int)
        segmented_predictions = (self.segmented_probabilities > 0.5).astype(int)

        # Accuracy
        full_accuracy = np.mean(full_predictions == self.test_labels)
        segmented_accuracy = np.mean(segmented_predictions == self.test_labels)

        # True Positives, False Positives, True Negatives, False Negatives
        full_tp = np.sum((full_predictions == 1) & (self.test_labels == 1))
        full_fp = np.sum((full_predictions == 1) & (self.test_labels == 0))
        full_tn = np.sum((full_predictions == 0) & (self.test_labels == 0))
        full_fn = np.sum((full_predictions == 0) & (self.test_labels == 1))

        segmented_tp = np.sum((segmented_predictions == 1) & (self.test_labels == 1))
        segmented_fp = np.sum((segmented_predictions == 1) & (self.test_labels == 0))
        segmented_tn = np.sum((segmented_predictions == 0) & (self.test_labels == 0))
        segmented_fn = np.sum((segmented_predictions == 0) & (self.test_labels == 1))

        # Precision, Recall, F1
        full_precision = full_tp / (full_tp + full_fp) if (full_tp + full_fp) > 0 else 0
        full_recall = full_tp / (full_tp + full_fn) if (full_tp + full_fn) > 0 else 0
        full_f1 = 2 * (full_precision * full_recall) / (full_precision + full_recall) if (
                                                                                                     full_precision + full_recall) > 0 else 0

        segmented_precision = segmented_tp / (segmented_tp + segmented_fp) if (segmented_tp + segmented_fp) > 0 else 0
        segmented_recall = segmented_tp / (segmented_tp + segmented_fn) if (segmented_tp + segmented_fn) > 0 else 0
        segmented_f1 = 2 * (segmented_precision * segmented_recall) / (segmented_precision + segmented_recall) if (
                                                                                                                              segmented_precision + segmented_recall) > 0 else 0

        # Calculate AUC-ROC
        full_auc = self.roc_curve_and_auc(self.test_labels, self.full_probabilities)
        segmented_auc = self.roc_curve_and_auc(self.test_labels, self.segmented_probabilities)

        # Store all metrics in a dictionary
        metrics = {
            'accuracy': {
                'full': float(full_accuracy),
                'segmented': float(segmented_accuracy),
                'difference': float(segmented_accuracy - full_accuracy),
                'percent_change': float(
                    (segmented_accuracy - full_accuracy) / full_accuracy * 100 if full_accuracy > 0 else 0)
            },
            'precision': {
                'full': float(full_precision),
                'segmented': float(segmented_precision),
                'difference': float(segmented_precision - full_precision),
                'percent_change': float(
                    (segmented_precision - full_precision) / full_precision * 100 if full_precision > 0 else 0)
            },
            'recall': {
                'full': float(full_recall),
                'segmented': float(segmented_recall),
                'difference': float(segmented_recall - full_recall),
                'percent_change': float((segmented_recall - full_recall) / full_recall * 100 if full_recall > 0 else 0)
            },
            'f1_score': {
                'full': float(full_f1),
                'segmented': float(segmented_f1),
                'difference': float(segmented_f1 - full_f1),
                'percent_change': float((segmented_f1 - full_f1) / full_f1 * 100 if full_f1 > 0 else 0)
            },
            'auc': {
                'full': float(full_auc['auc']),
                'segmented': float(segmented_auc['auc']),
                'difference': float(segmented_auc['auc'] - full_auc['auc']),
                'percent_change': float(
                    (segmented_auc['auc'] - full_auc['auc']) / full_auc['auc'] * 100 if full_auc['auc'] > 0 else 0)
            },
            'confusion_matrix': {
                'full': {
                    'tp': int(full_tp),
                    'fp': int(full_fp),
                    'tn': int(full_tn),
                    'fn': int(full_fn)
                },
                'segmented': {
                    'tp': int(segmented_tp),
                    'fp': int(segmented_fp),
                    'tn': int(segmented_tn),
                    'fn': int(segmented_fn)
                }
            }
        }

        # Also analyze predictions that differ between models
        different_predictions = np.where(full_predictions != segmented_predictions)[0]
        print(f"Found {len(different_predictions)} samples with different predictions between models")

        # Count where each model is correct when they disagree
        full_correct_in_diff = np.sum(
            full_predictions[different_predictions] == self.test_labels[different_predictions])
        segmented_correct_in_diff = np.sum(
            segmented_predictions[different_predictions] == self.test_labels[different_predictions])

        metrics['divergent_predictions'] = {
            'count': int(len(different_predictions)),
            'percentage': float(len(different_predictions) / len(self.test_labels) * 100),
            'full_correct': int(full_correct_in_diff),
            'segmented_correct': int(segmented_correct_in_diff),
            'full_correct_percent': float(
                full_correct_in_diff / len(different_predictions) * 100 if len(different_predictions) > 0 else 0),
            'segmented_correct_percent': float(
                segmented_correct_in_diff / len(different_predictions) * 100 if len(different_predictions) > 0 else 0)
        }

        # Store reference to metrics
        self.performance_metrics = metrics

        # Save metrics to JSON
        with open(os.path.join(self.output_dir, 'performance_comparison.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Print summary of metrics
        print("\nPerformance Metrics Comparison:")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            metric = metrics[metric_name]
            print(f"{metric_name.capitalize()}: Full={metric['full']:.4f}, "
                  f"Segmented={metric['segmented']:.4f}, "
                  f"Diff={metric['difference']:.4f} ({metric['percent_change']:+.2f}%)")

        print(f"\nDivergent predictions: {metrics['divergent_predictions']['count']} "
              f"({metrics['divergent_predictions']['percentage']:.2f}% of test set)")
        print(f"  Full model correct: {metrics['divergent_predictions']['full_correct_percent']:.2f}%")
        print(f"  Segmented model correct: {metrics['divergent_predictions']['segmented_correct_percent']:.2f}%")

        # Create visualizations
        self.visualize_performance_metrics(metrics, full_auc, segmented_auc)

        return metrics

    def roc_curve_and_auc(self, y_true, y_prob):
        """
        Calculate ROC curve and AUC.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities

        Returns:
            dict: Dictionary with fpr, tpr, thresholds, and auc
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)

        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(auc_score)
        }

    def visualize_performance_metrics(self, metrics, full_auc_data, segmented_auc_data):
        """
        Create visualizations comparing performance metrics between models.

        Args:
            metrics: Dictionary of performance metrics
            full_auc_data: ROC curve data for full model
            segmented_auc_data: ROC curve data for segmented model
        """
        # Create output directory for metric plots
        metric_plots_dir = os.path.join(self.plots_dir, 'metrics')
        os.makedirs(metric_plots_dir, exist_ok=True)

        # Bar chart comparing key metrics
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        full_values = [metrics[m]['full'] for m in metric_names]
        segmented_values = [metrics[m]['segmented'] for m in metric_names]

        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        x = np.arange(len(metric_names))

        plt.bar(x - bar_width / 2, full_values, bar_width, label='Full Image Model')
        plt.bar(x + bar_width / 2, segmented_values, bar_width, label='Segmented Model')

        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Performance Metrics Comparison')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metric_names])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Add value labels above bars
        for i, v in enumerate(full_values):
            plt.text(i - bar_width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(segmented_values):
            plt.text(i + bar_width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(metric_plots_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()

        # Plot ROC curves
        plt.figure(figsize=(10, 8))

        # Full model
        plt.plot(full_auc_data['fpr'], full_auc_data['tpr'],
                 label=f'Full Image (AUC = {full_auc_data["auc"]:.4f})',
                 linewidth=2)

        # Segmented model
        plt.plot(segmented_auc_data['fpr'], segmented_auc_data['tpr'],
                 label=f'Segmented (AUC = {segmented_auc_data["auc"]:.4f})',
                 linewidth=2)

        # Reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(metric_plots_dir, 'roc_curve_comparison.png'), dpi=300)
        plt.close()

        # Visualize confusion matrices
        self.plot_confusion_matrices(metrics['confusion_matrix'])

    def plot_confusion_matrices(self, cm_data):
        """
        Plot and compare confusion matrices between models.

        Args:
            cm_data: Dictionary with confusion matrix data
        """
        # Create figure with two subplots for the confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Prepare confusion matrices as 2x2 arrays
        full_cm = np.array([[cm_data['full']['tn'], cm_data['full']['fp']],
                            [cm_data['full']['fn'], cm_data['full']['tp']]])

        segmented_cm = np.array([[cm_data['segmented']['tn'], cm_data['segmented']['fp']],
                                 [cm_data['segmented']['fn'], cm_data['segmented']['tp']]])

        # Normalize by row (true labels)
        full_cm_norm = full_cm.astype('float') / full_cm.sum(axis=1)[:, np.newaxis]
        segmented_cm_norm = segmented_cm.astype('float') / segmented_cm.sum(axis=1)[:, np.newaxis]

        # Plot full model confusion matrix
        sns.heatmap(full_cm_norm, annot=full_cm, fmt='d', cmap='Blues',
                    cbar=False, ax=ax1, annot_kws={"size": 16})
        ax1.set_title('Full Image Model')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_xticklabels(['Benign', 'Malware'])
        ax1.set_yticklabels(['Benign', 'Malware'])

        # Plot segmented model confusion matrix
        sns.heatmap(segmented_cm_norm, annot=segmented_cm, fmt='d', cmap='Blues',
                    cbar=False, ax=ax2, annot_kws={"size": 16})
        ax2.set_title('Segmented Model')
        ax2.set_xlabel('Predicted Label')
        ax2.set_xticklabels(['Benign', 'Malware'])
        ax2.set_yticklabels(['Benign', 'Malware'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'metrics', 'confusion_matrices.png'), dpi=300)
        plt.close()

        # Also create a difference heatmap
        plt.figure(figsize=(8, 6))

        # Calculate the differences (segmented - full)
        diff_cm = segmented_cm - full_cm

        # Custom colormap centered at zero
        cmap = plt.cm.RdBu_r

        # Create heatmap
        sns.heatmap(diff_cm, annot=True, fmt='d', cmap=cmap, center=0,
                    cbar_kws={'label': 'Difference (Segmented - Full)'})
        plt.title('Confusion Matrix Difference (Segmented - Full)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks([0.5, 1.5], ['Benign', 'Malware'])
        plt.yticks([0.5, 1.5], ['Benign', 'Malware'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'metrics', 'confusion_matrix_difference.png'), dpi=300)
        plt.close()

    def analyze_prediction_differences(self):
        """
        Analyze samples where the two models make different predictions.
        """
        print("Analyzing prediction differences between models...")

        # Ensure data is loaded
        if self.full_probabilities is None or self.segmented_probabilities is None:
            if not self.load_data():
                print("Failed to load data. Cannot analyze prediction differences.")
                return

        # Get binary predictions
        full_predictions = (self.full_probabilities > 0.5).astype(int)
        segmented_predictions = (self.segmented_probabilities > 0.5).astype(int)

        # Find samples where predictions differ
        diff_indices = np.where(full_predictions != segmented_predictions)[0]

        if len(diff_indices) == 0:
            print("No prediction differences found.")
            return

        print(
            f"Found {len(diff_indices)} samples with different predictions ({len(diff_indices) / len(full_predictions) * 100:.2f}% of test set)")

        # Analyze differences
        analysis_data = []

        for idx in diff_indices:
            sample_data = {
                'index': int(idx),
                'file_path': str(self.test_files[idx]) if self.test_files is not None else f"sample_{idx}",
                'true_label': int(self.test_labels[idx]) if self.test_labels is not None else None,
                'full_pred': int(full_predictions[idx]),
                'segmented_pred': int(segmented_predictions[idx]),
                'full_prob': float(self.full_probabilities[idx]),
                'segmented_prob': float(self.segmented_probabilities[idx]),
                'prob_diff': float(self.segmented_probabilities[idx] - self.full_probabilities[idx]),
                'correct_model': None
            }

            # Determine which model is correct (if we have ground truth)
            if self.test_labels is not None:
                true_label = self.test_labels[idx]

                if full_predictions[idx] == true_label and segmented_predictions[idx] != true_label:
                    sample_data['correct_model'] = 'full'
                elif full_predictions[idx] != true_label and segmented_predictions[idx] == true_label:
                    sample_data['correct_model'] = 'segmented'
                else:
                    sample_data['correct_model'] = 'neither'  # Should not happen if predictions differ

            # Add family information if available
            if self.family_labels is not None:
                sample_data['family'] = self.family_labels[idx]
            elif hasattr(self, 'file_family_map') and self.test_files is not None:
                sample_data['family'] = self.file_family_map.get(str(self.test_files[idx]), 'Unknown')

            analysis_data.append(sample_data)

        # Convert to DataFrame for easier analysis
        diff_df = pd.DataFrame(analysis_data)

        # Save differences to CSV
        diff_csv_path = os.path.join(self.output_dir, 'prediction_differences.csv')
        diff_df.to_csv(diff_csv_path, index=False)
        print(f"Saved prediction differences to {diff_csv_path}")

        # Create summary statistics
        diff_summary = {
            'total_differences': len(diff_indices),
            'percent_of_test_set': float(len(diff_indices) / len(full_predictions) * 100)
        }

        if 'correct_model' in diff_df.columns:
            # Count correct predictions by each model
            model_correctness = diff_df['correct_model'].value_counts().to_dict()

            diff_summary['full_correct'] = model_correctness.get('full', 0)
            diff_summary['segmented_correct'] = model_correctness.get('segmented', 0)
            diff_summary['neither_correct'] = model_correctness.get('neither', 0)

            diff_summary['full_correct_percent'] = diff_summary['full_correct'] / len(diff_indices) * 100
            diff_summary['segmented_correct_percent'] = diff_summary['segmented_correct'] / len(diff_indices) * 100

            # Print summary
            print(f"Among differing predictions:")
            print(f"  Full model correct: {diff_summary['full_correct']} ({diff_summary['full_correct_percent']:.2f}%)")
            print(
                f"  Segmented model correct: {diff_summary['segmented_correct']} ({diff_summary['segmented_correct_percent']:.2f}%)")

            if 'family' in diff_df.columns:
                # Analyze by family
                family_analysis = pd.crosstab(
                    diff_df['family'],
                    diff_df['correct_model'],
                    normalize='index'
                ) * 100

                # Add sample counts
                family_counts = diff_df['family'].value_counts()
                family_analysis['sample_count'] = family_counts

                # Save family-level analysis
                family_csv_path = os.path.join(self.family_dir, 'family_prediction_differences.csv')
                family_analysis.to_csv(family_csv_path)

                # Visualize
                if 'sample_count' in family_analysis.columns:
                    self.visualize_family_prediction_differences(family_analysis)

        # Visualize difference distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(diff_df['prob_diff'], bins=50, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Probability Difference (Segmented - Full)')
        plt.ylabel('Count')
        plt.title('Distribution of Probability Differences in Divergent Predictions')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'probability_difference_distribution.png'), dpi=300)
        plt.close()

        # Save summary
        with open(os.path.join(self.output_dir, 'prediction_differences_summary.json'), 'w') as f:
            json.dump(diff_summary, f, indent=4)

        return diff_summary

    def visualize_family_prediction_differences(self, family_analysis):
        """
        Visualize prediction differences by family.

        Args:
            family_analysis: DataFrame with family-level analysis
        """
        # Sort by sample count for better visualization
        family_analysis = family_analysis.sort_values('sample_count', ascending=False)

        if len(family_analysis) > 15:
            # If too many families, only show top 15 by sample count
            family_analysis = family_analysis.head(15)

        # Plot stacked bar chart of correct predictions by family
        plt.figure(figsize=(12, 8))

        # Prepare the data for plotting
        if 'full' in family_analysis.columns and 'segmented' in family_analysis.columns:
            plot_data = family_analysis[['full', 'segmented']].copy()

            # Convert to percentages
            for col in plot_data.columns:
                if plot_data[col].dtype != 'object':
                    plot_data[col] = plot_data[col].astype(float)

            # Plot
            plot_data.plot(kind='bar', stacked=False, figsize=(12, 8),
                           color=['blue', 'green'], alpha=0.7)

            plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
            plt.title('Model Correctness by Family (Where Predictions Differ)')
            plt.xlabel('Malware Family')
            plt.ylabel('% Correct Predictions')
            plt.legend(['50% Threshold', 'Full Image Model', 'Segmented Model'])
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)

            # Add sample count annotations
            for i, (_, row) in enumerate(family_analysis.iterrows()):
                plt.text(i, 105, f"n={int(row['sample_count'])}", ha='center', va='bottom',
                         rotation=90, fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(self.family_dir, 'correctness_by_family.png'), dpi=300)
            plt.close()

    def analyze_confidence_calibration(self):
        """
        Analyze and compare confidence calibration between the models.
        """
        print("Analyzing confidence calibration...")

        # Ensure data is loaded
        if self.full_probabilities is None or self.segmented_probabilities is None:
            if not self.load_data():
                print("Failed to load data. Cannot analyze confidence calibration.")
                return

        # Create calibration plots
        n_bins = 10

        # Create binned predictions for full model
        full_bin_data = self.bin_predictions(self.full_probabilities, self.test_labels, n_bins)

        # Create binned predictions for segmented model
        segmented_bin_data = self.bin_predictions(self.segmented_probabilities, self.test_labels, n_bins)

        # Calculate ECE (Expected Calibration Error)
        full_ece = self.expected_calibration_error(full_bin_data)
        segmented_ece = self.expected_calibration_error(segmented_bin_data)

        print(f"Expected Calibration Error (ECE):")
        print(f"  Full model: {full_ece:.4f}")
        print(f"  Segmented model: {segmented_ece:.4f}")

        # Save calibration results
        calibration_results = {
            'full_model': {
                'bin_data': full_bin_data,
                'ece': float(full_ece)
            },
            'segmented_model': {
                'bin_data': segmented_bin_data,
                'ece': float(segmented_ece)
            },
            'ece_difference': float(segmented_ece - full_ece),
            'better_calibrated': 'full' if full_ece < segmented_ece else 'segmented'
        }

        with open(os.path.join(self.output_dir, 'calibration_results.json'), 'w') as f:
            json.dump(calibration_results, f, indent=4)

        # Create calibration plots
        self.plot_calibration_curves(full_bin_data, segmented_bin_data, full_ece, segmented_ece)

        return calibration_results

    def bin_predictions(self, probs, labels, n_bins=10):
        """
        Bin predictions for calibration analysis.

        Args:
            probs: Predicted probabilities
            labels: True labels
            n_bins: Number of bins to use

        Returns:
            list: List of dictionaries with bin data
        """
        # Create bins
        bin_size = 1.0 / n_bins
        bins = np.linspace(0.0, 1.0, n_bins + 1)

        bin_data = []

        for i in range(n_bins):
            # Find predictions in this bin
            bin_start = bins[i]
            bin_end = bins[i + 1]

            # For the last bin, include 1.0
            if i == n_bins - 1:
                indices = np.where((probs >= bin_start) & (probs <= bin_end))[0]
            else:
                indices = np.where((probs >= bin_start) & (probs < bin_end))[0]

            if len(indices) > 0:
                # Calculate statistics for this bin
                bin_probs = probs[indices]
                bin_labels = labels[indices]

                bin_confidence = np.mean(bin_probs)
                bin_accuracy = np.mean(bin_labels)
                bin_samples = len(indices)

                bin_data.append({
                    'bin_start': float(bin_start),
                    'bin_end': float(bin_end),
                    'bin_mid': float(bin_start + bin_size / 2),
                    'confidence': float(bin_confidence),
                    'accuracy': float(bin_accuracy),
                    'samples': int(bin_samples),
                    'fraction': float(bin_samples / len(probs))
                })

        return bin_data

    def expected_calibration_error(self, bin_data):
        """
        Calculate the Expected Calibration Error.

        Args:
            bin_data: List of dictionaries with bin statistics

        Returns:
            float: Expected Calibration Error
        """
        ece = 0.0
        total_samples = sum(bin_info['samples'] for bin_info in bin_data)

        for bin_info in bin_data:
            bin_weight = bin_info['samples'] / total_samples
            bin_calibration_error = abs(bin_info['confidence'] - bin_info['accuracy'])
            ece += bin_weight * bin_calibration_error

        return ece

    def plot_calibration_curves(self, full_bin_data, segmented_bin_data, full_ece, segmented_ece):
        """
        Create calibration plots comparing both models.

        Args:
            full_bin_data: Bin data for full model
            segmented_bin_data: Bin data for segmented model
            full_ece: Expected Calibration Error for full model
            segmented_ece: Expected Calibration Error for segmented model
        """
        # Extract data for plotting
        full_conf = [b['confidence'] for b in full_bin_data]
        full_acc = [b['accuracy'] for b in full_bin_data]
        full_samples = [b['samples'] for b in full_bin_data]
        full_mid = [b['bin_mid'] for b in full_bin_data]

        segmented_conf = [b['confidence'] for b in segmented_bin_data]
        segmented_acc = [b['accuracy'] for b in segmented_bin_data]
        segmented_samples = [b['samples'] for b in segmented_bin_data]
        segmented_mid = [b['bin_mid'] for b in segmented_bin_data]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Perfect calibration reference line
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        # Full model calibration
        ax1.plot(full_conf, full_acc, marker='o', linestyle='-',
                 label=f'Full Model (ECE={full_ece:.4f})')

        # Size points by bin count
        for i, (x, y, n) in enumerate(zip(full_conf, full_acc, full_samples)):
            ax1.scatter([x], [y], s=n / 30, alpha=0.6, color='blue')

        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Calibration Curve - Full Image Model')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)

        # Segmented model calibration
        ax2.plot(segmented_conf, segmented_acc, marker='o', linestyle='-',
                 label=f'Segmented Model (ECE={segmented_ece:.4f})')

        # Size points by bin count
        for i, (x, y, n) in enumerate(zip(segmented_conf, segmented_acc, segmented_samples)):
            ax2.scatter([x], [y], s=n / 30, alpha=0.6, color='blue')

        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Calibration Curve - Segmented Model')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'calibration_curves.png'), dpi=300)
        plt.close()

        # Create combined reliability diagram
        plt.figure(figsize=(10, 8))

        # Perfect calibration reference line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        # Full model
        plt.plot(full_mid, full_acc, marker='o', linestyle='-',
                 label=f'Full Model (ECE={full_ece:.4f})')

        # Segmented model
        plt.plot(segmented_mid, segmented_acc, marker='s', linestyle='-',
                 label=f'Segmented Model (ECE={segmented_ece:.4f})')

        plt.xlabel('Predicted Probability (Bin Center)')
        plt.ylabel('Observed Accuracy')
        plt.title('Reliability Diagram - Both Models')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'reliability_diagram.png'), dpi=300)
        plt.close()

    def run_analysis(self):
        """
        Run the complete statistical analysis pipeline.
        """
        print("\n=== Starting Statistical Analysis ===")
        print(f"Full model results from: {self.full_model_dir}")
        print(f"Segmented model results from: {self.segmented_model_dir}")
        print(f"Saving output to: {self.output_dir}")

        # Load data
        if not self.load_data():
            print("Failed to load required data. Analysis aborted.")
            return False

        # Perform KS test
        self.perform_ks_test()

        # Compare probability distributions
        self.compare_probability_distributions()

        # Compare performance metrics
        self.compare_performance_metrics()

        # Analyze prediction differences
        self.analyze_prediction_differences()

        # Family-level analysis
        if self.family_labels is not None or self.file_family_map:
            self.family_level_analysis()

        # Confidence calibration analysis
        self.analyze_confidence_calibration()

        print("\n=== Analysis Complete ===")
        print(f"Results saved to {self.output_dir}")
        return True

    def create_summary_report(self):
        """
        Create a summary report of all findings.
        """
        # Only create report if all necessary data is available
        if not hasattr(self, 'ks_results') or not hasattr(self, 'performance_metrics'):
            print("Cannot create summary report - analysis incomplete")
            return

        print("Creating summary report...")

        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'full_model_dir': self.full_model_dir,
            'segmented_model_dir': self.segmented_model_dir,
            'output_dir': self.output_dir,
            'sample_size': len(self.full_probabilities),
            'ks_test': self.ks_results,
            'performance_comparison': self.performance_metrics
        }

        # Add family level analysis if available
        if hasattr(self, 'per_family_results') and self.per_family_results:
            report['family_analysis'] = self.per_family_results

        # Determine overall finding
        if self.ks_results.get('significant', False):
            report[
                'distribution_finding'] = "The probability distributions from the full image and segmented models are significantly different."
        else:
            report[
                'distribution_finding'] = "No significant difference detected between probability distributions from the two models."

        # Performance comparison summary
        better_metrics = []
        for metric, values in self.performance_metrics.items():
            if metric not in ['confusion_matrix', 'divergent_predictions']:
                if values.get('difference', 0) > 0:
                    better_metrics.append(metric)

        if len(better_metrics) > len(self.performance_metrics) / 2:
            report['performance_finding'] = "The segmented model shows better performance across most metrics."
        elif len(better_metrics) < len(self.performance_metrics) / 4:
            report['performance_finding'] = "The full image model shows better performance across most metrics."
        else:
            report['performance_finding'] = "The models show mixed performance advantages across different metrics."

        # Save report
        with open(os.path.join(self.output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(report, f, indent=4)

        # Create text summary
        summary_text = []
        summary_text.append("# Statistical Analysis Summary")
        summary_text.append(f"Analysis performed on {report['timestamp']}")
        summary_text.append(f"\n## Kolmogorov-Smirnov Test Results")
        summary_text.append(f"- KS statistic: {self.ks_results['statistic']:.4f}")
        summary_text.append(f"- p-value: {self.ks_results['p_value']:.4g}")
        summary_text.append(f"- Significant: {self.ks_results['significant']}")
        summary_text.append(f"- Finding: {report['distribution_finding']}")

        summary_text.append(f"\n## Performance Metrics Comparison")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            values = self.performance_metrics[metric]
            diff = values['difference']
            pct = values['percent_change']

            if diff > 0:
                better = "segmented"
                emoji = "✅"
            else:
                better = "full image"
                emoji = "❌"

            summary_text.append(
                f"- {metric.title()}: Full={values['full']:.4f}, Segmented={values['segmented']:.4f}, " +
                f"Diff={diff:+.4f} ({pct:+.2f}%) - {emoji} {better} model better")

        # Add divergent predictions
        div_preds = self.performance_metrics.get('divergent_predictions', {})
        if div_preds:
            summary_text.append(f"\n## Prediction Differences")
            summary_text.append(
                f"- {div_preds['count']} samples ({div_preds['percentage']:.2f}% of test set) have different predictions")
            summary_text.append(f"- When predictions differ:")
            summary_text.append(f"  - Full model correct: {div_preds['full_correct_percent']:.2f}%")
            summary_text.append(f"  - Segmented model correct: {div_preds['segmented_correct_percent']:.2f}%")

        # Add family analysis summary if available
        if 'family_analysis' in report:
            summary_text.append(f"\n## Family-Level Analysis")
            summary_text.append(f"- Analysis performed across {len(report['family_analysis'])} malware families")

            # Find families where one model clearly outperforms
            full_better = []
            segmented_better = []

            for family, data in report['family_analysis'].items():
                if data.get('accuracy_diff', 0) > 0.05:  # 5% threshold
                    segmented_better.append(family)
                elif data.get('accuracy_diff', 0) < -0.05:
                    full_better.append(family)

            if segmented_better:
                summary_text.append(
                    f"- Families where segmented model clearly performs better: {', '.join(segmented_better)}")
            if full_better:
                summary_text.append(
                    f"- Families where full image model clearly performs better: {', '.join(full_better)}")

        # Overall conclusion
        summary_text.append(f"\n## Conclusion")
        summary_text.append(report['performance_finding'])

        if 'distribution_finding' in report:
            summary_text.append(report['distribution_finding'])

        # Save text summary
        with open(os.path.join(self.output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write('\n'.join(summary_text))

        print(f"Summary report saved to {os.path.join(self.output_dir, 'analysis_summary.txt')}")
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Statistical Analysis for Malware Classification Models')

    parser.add_argument('--full-model-dir', type=str, required=True,
                        help='Directory containing full image model results')
    parser.add_argument('--segmented-model-dir', type=str, required=True,
                        help='Directory containing segmented model results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis results')

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Initialize the analyzer
    analyzer = StatisticalAnalyzer(
        full_model_dir=args.full_model_dir,
        segmented_model_dir=args.segmented_model_dir,
        output_dir=args.output_dir
    )

    # Run the analysis
    analyzer.run_analysis()

    # Create summary report
    analyzer.create_summary_report()

    return 0


if __name__ == "__main__":
    main()