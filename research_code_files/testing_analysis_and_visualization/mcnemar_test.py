#!/usr/bin/env python3
"""
McNemar's Test Module for Malware Classification Model Comparison

This module performs statistical analysis to determine if there is a significant
difference in predictive performance between two models:
1. Full-image CNN model (256x256)
2. Segment-based CNN model (128x128 high-variance segments)

The test specifically uses McNemar's test, which is appropriate for comparing
two models on the same test set, focusing on the disagreements between models.
"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress the deprecated binom_test warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message="'binom_test' is deprecated")
import argparse


class ModelComparator:
    def __init__(self, full_model_dir, segment_model_dir, output_dir=None):
        """
        Initialize the model comparator.

        Args:
            full_model_dir: Directory containing full-image model results
            segment_model_dir: Directory containing segment-based model results
            output_dir: Directory to save comparison results (defaults to a subdirectory of segment_model_dir)
        """
        self.full_model_dir = full_model_dir
        self.segment_model_dir = segment_model_dir

        # Default output directory is a subdirectory of segment_model_dir
        if output_dir is None:
            self.output_dir = os.path.join(segment_model_dir, 'comparison_results')
        else:
            self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize data containers
        self.full_model_preds = None
        self.full_model_probs = None
        self.segment_model_preds = None
        self.segment_model_probs = None
        self.true_labels = None
        self.test_files = None
        self.segment_metadata = None
        self.comparison_results = {}

    def load_data(self):
        """
        Load prediction results and test data from both models.

        Returns:
            Boolean indicating if data loading was successful
        """
        try:
            # Load full model predictions
            self.full_model_probs = np.load(os.path.join(self.full_model_dir, 'metrics', 'y_pred_prob.npy')).flatten()
            self.full_model_preds = (self.full_model_probs > 0.5).astype(int)

            # Load true labels
            self.true_labels = np.load(os.path.join(self.full_model_dir, 'metrics', 'y_true.npy'))

            # Load test file paths
            test_file_paths_path = os.path.join(self.full_model_dir, 'metrics', 'test_file_paths.npy')
            if os.path.exists(test_file_paths_path):
                # Convert file paths to strings to avoid JSON serialization issues
                self.test_files = np.load(test_file_paths_path, allow_pickle=True)
                self.test_files = np.array([str(f) for f in self.test_files])

            # Load segment model predictions
            self.segment_model_probs = np.load(os.path.join(self.segment_model_dir, 'test_probabilities.npy')).flatten()
            self.segment_model_preds = np.load(os.path.join(self.segment_model_dir, 'test_predictions.npy')).flatten()

            # Load segment metadata if available
            segment_metadata_path = os.path.join(self.segment_model_dir, 'segment_selection_metadata.json')
            if os.path.exists(segment_metadata_path):
                with open(segment_metadata_path, 'r') as f:
                    self.segment_metadata = json.load(f)

            # Validate shapes
            if len(self.full_model_preds) != len(self.segment_model_preds) or len(self.full_model_preds) != len(
                    self.true_labels):
                print("WARNING: Mismatched prediction array lengths!")
                print(f"Full model predictions: {len(self.full_model_preds)}")
                print(f"Segment model predictions: {len(self.segment_model_preds)}")
                print(f"True labels: {len(self.true_labels)}")
                return False

            print(f"Successfully loaded predictions for {len(self.true_labels)} test samples")
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def perform_mcnemars_test(self):
        """
        Perform McNemar's test to compare the two models.

        Returns:
            Dictionary containing test results
        """
        # Create contingency table
        # b: full model correct, segment model incorrect
        # c: full model incorrect, segment model correct
        b = np.sum((self.full_model_preds == self.true_labels) &
                   (self.segment_model_preds != self.true_labels))

        c = np.sum((self.full_model_preds != self.true_labels) &
                   (self.segment_model_preds == self.true_labels))

        # Also calculate the concordant cases for completeness
        a = np.sum((self.full_model_preds == self.true_labels) &
                   (self.segment_model_preds == self.true_labels))

        d = np.sum((self.full_model_preds != self.true_labels) &
                   (self.segment_model_preds != self.true_labels))

        # Create the contingency table
        contingency_table = np.array([[a, b], [c, d]])

        # Perform McNemar's test
        # Use Edwards' correction for continuity if b + c < 25
        if b + c < 25:
            statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        else:
            statistic = ((b - c) ** 2) / (b + c)

        p_value = stats.chi2.sf(statistic, 1)  # Chi-square with 1 degree of freedom

        # Calculate exact binomial test p-value as well (more accurate for small b+c)
        if b + c > 0:
            exact_p_value = stats.binom_test(min(b, c), b + c, p=0.5)
        else:
            exact_p_value = 1.0

        # Store and return results
        self.comparison_results = {
            'contingency_table': contingency_table.tolist(),
            'both_correct': int(a),
            'full_correct_segment_wrong': int(b),
            'full_wrong_segment_correct': int(c),
            'both_wrong': int(d),
            'total_samples': int(a + b + c + d),
            'mcnemar_statistic': float(statistic),
            'mcnemar_p_value': float(p_value),
            'exact_p_value': float(exact_p_value),
            'is_significant_005': bool(p_value < 0.05),
            'better_model': 'full_image' if b > c else 'segment' if c > b else 'tie'
        }

        return self.comparison_results

    def calculate_model_metrics(self):
        """
        Calculate and compare performance metrics for both models.

        Returns:
            Dictionary containing metric comparisons
        """
        # Performance metrics for full model
        full_metrics = {
            'accuracy': accuracy_score(self.true_labels, self.full_model_preds),
            'precision': precision_score(self.true_labels, self.full_model_preds, zero_division=0),
            'recall': recall_score(self.true_labels, self.full_model_preds, zero_division=0),
            'f1': f1_score(self.true_labels, self.full_model_preds, zero_division=0)
        }

        # Performance metrics for segment model
        segment_metrics = {
            'accuracy': accuracy_score(self.true_labels, self.segment_model_preds),
            'precision': precision_score(self.true_labels, self.segment_model_preds, zero_division=0),
            'recall': recall_score(self.true_labels, self.segment_model_preds, zero_division=0),
            'f1': f1_score(self.true_labels, self.segment_model_preds, zero_division=0)
        }

        # Calculate differences
        metric_diff = {
            metric: segment_metrics[metric] - full_metrics[metric]
            for metric in full_metrics
        }

        # Store in comparison results
        self.comparison_results['metrics'] = {
            'full_model': full_metrics,
            'segment_model': segment_metrics,
            'difference': metric_diff
        }

        return self.comparison_results['metrics']

    def analyze_disagreements(self):
        """
        Analyze the samples where the models disagree.

        Returns:
            DataFrame containing detailed disagreement analysis
        """
        # Get indices where models disagree
        disagreement_idx = np.where(self.full_model_preds != self.segment_model_preds)[0]

        # If no disagreements, return empty dataframe
        if len(disagreement_idx) == 0:
            return pd.DataFrame()

        # Create dataframe to analyze disagreements
        disagreements = []

        for idx in disagreement_idx:
            is_full_correct = self.full_model_preds[idx] == self.true_labels[idx]
            is_segment_correct = self.segment_model_preds[idx] == self.true_labels[idx]

            # Get file path if available
            file_path = self.test_files[idx] if self.test_files is not None else f"sample_{idx}"

            # Get segment metadata if available
            segment_info = None
            if self.segment_metadata is not None and 'test' in self.segment_metadata:
                # Find matching file in segment metadata
                for entry in self.segment_metadata['test']:
                    if entry['file_path'] == file_path:
                        segment_info = entry
                        break

            # Add to disagreements list
            disagreement = {
                'index': idx,
                'file_path': file_path,
                'true_label': int(self.true_labels[idx]),
                'full_model_pred': int(self.full_model_preds[idx]),
                'segment_model_pred': int(self.segment_model_preds[idx]),
                'full_model_prob': float(self.full_model_probs[idx]),
                'segment_model_prob': float(self.segment_model_probs[idx]),
                'full_model_correct': is_full_correct,
                'segment_model_correct': is_segment_correct,
                'correct_model': 'full' if is_full_correct else 'segment' if is_segment_correct else 'neither'
            }

            # Add segment info if available
            if segment_info is not None:
                disagreement['selected_segment'] = segment_info['selected_segment']
                disagreement['segment_variances'] = segment_info['segment_variances']
                if 'family' in segment_info:
                    disagreement['family'] = segment_info['family']

            disagreements.append(disagreement)

        # Convert to DataFrame
        disagreements_df = pd.DataFrame(disagreements)

        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'model_disagreements.csv')
        disagreements_df.to_csv(csv_path, index=False)
        print(f"Saved detailed disagreement analysis to {csv_path}")

        # Add summary to comparison results
        self.comparison_results['disagreement_analysis'] = {
            'total_disagreements': len(disagreements),
            'full_model_correct': sum(d['full_model_correct'] for d in disagreements),
            'segment_model_correct': sum(d['segment_model_correct'] for d in disagreements),
            'neither_correct': sum(
                not d['full_model_correct'] and not d['segment_model_correct'] for d in disagreements)
        }

        return disagreements_df

    def analyze_by_family(self):
        """
        Analyze model performance by malware family.

        Returns:
            Dictionary with per-family performance comparison
        """
        # Check if family information is available
        if self.segment_metadata is None or 'test' not in self.segment_metadata:
            print("Family information not available, skipping family analysis")
            return None

        # Extract family information
        test_entries = self.segment_metadata['test']
        family_info = []

        for i, entry in enumerate(test_entries):
            if i < len(self.true_labels):
                family_info.append({
                    'file_path': entry['file_path'],
                    'family': entry['family'],
                    'true_label': int(self.true_labels[i]),
                    'full_model_pred': int(self.full_model_preds[i]),
                    'segment_model_pred': int(self.segment_model_preds[i])
                })

        # Create DataFrame
        family_df = pd.DataFrame(family_info)

        # Calculate per-family metrics
        family_analysis = {}

        for family in family_df['family'].unique():
            family_samples = family_df[family_df['family'] == family]
            true_labels = family_samples['true_label']
            full_preds = family_samples['full_model_pred']
            segment_preds = family_samples['segment_model_pred']

            # Skip families with too few samples (less than 5)
            if len(true_labels) < 5:
                continue

            # Calculate metrics
            full_acc = accuracy_score(true_labels, full_preds)
            segment_acc = accuracy_score(true_labels, segment_preds)

            # Calculate McNemar's test for this family
            # Only if there are disagreements
            b = np.sum((full_preds == true_labels) & (segment_preds != true_labels))
            c = np.sum((full_preds != true_labels) & (segment_preds == true_labels))

            if b + c > 0:
                # Use Edwards' correction for continuity if b + c < 25
                if b + c < 25:
                    statistic = ((abs(b - c) - 1) ** 2) / (b + c)
                else:
                    statistic = ((b - c) ** 2) / (b + c)

                p_value = stats.chi2.sf(statistic, 1)
            else:
                statistic = 0
                p_value = 1.0

            family_analysis[family] = {
                'sample_count': len(true_labels),
                'full_model_accuracy': full_acc,
                'segment_model_accuracy': segment_acc,
                'accuracy_diff': segment_acc - full_acc,
                'mcnemar_statistic': float(statistic),
                'mcnemar_p_value': float(p_value),
                'is_significant': p_value < 0.05,
                'better_model': 'full_image' if b > c else 'segment' if c > b else 'tie'
            }

        # Save to comparison results
        self.comparison_results['family_analysis'] = family_analysis

        return family_analysis

    def create_visualizations(self):
        """
        Create visualizations to illustrate the comparison results.
        """
        # 1. Contingency Table Visualization
        if 'contingency_table' in self.comparison_results:
            contingency = np.array(self.comparison_results['contingency_table'])

            plt.figure(figsize=(10, 8))
            sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Segment Correct', 'Segment Wrong'],
                        yticklabels=['Full Correct', 'Full Wrong'])
            plt.title('McNemar\'s Contingency Table')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'contingency_table.png'), dpi=300)
            plt.close()

        # 2. Performance Metrics Comparison
        if 'metrics' in self.comparison_results:
            metrics = self.comparison_results['metrics']

            # Create a bar chart comparing metrics
            metrics_df = pd.DataFrame({
                'Full Image Model': [metrics['full_model'][m] for m in ['accuracy', 'precision', 'recall', 'f1']],
                'Segment Model': [metrics['segment_model'][m] for m in ['accuracy', 'precision', 'recall', 'f1']]
            }, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

            plt.figure(figsize=(12, 8))
            metrics_df.plot(kind='bar', ax=plt.gca())
            plt.title('Performance Metrics Comparison')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), dpi=300)
            plt.close()

        # 3. Family-wise Performance Comparison (if available)
        if 'family_analysis' in self.comparison_results:
            family_analysis = self.comparison_results['family_analysis']

            # Convert to DataFrame for easier plotting
            family_data = []
            for family, data in family_analysis.items():
                family_data.append({
                    'Family': family,
                    'Sample Count': data['sample_count'],
                    'Full Model Accuracy': data['full_model_accuracy'],
                    'Segment Model Accuracy': data['segment_model_accuracy'],
                    'Accuracy Difference': data['accuracy_diff'],
                    'McNemar p-value': data['mcnemar_p_value'],
                    'Significant': data['is_significant'],
                    'Better Model': data['better_model']
                })

            family_df = pd.DataFrame(family_data)

            # Sort by absolute accuracy difference
            family_df['Abs Difference'] = family_df['Accuracy Difference'].abs()
            family_df = family_df.sort_values('Abs Difference', ascending=False)

            # Plot accuracy by family
            plt.figure(figsize=(14, 10))

            # Create grouped bar chart
            x = np.arange(len(family_df))
            width = 0.35

            plt.bar(x - width / 2, family_df['Full Model Accuracy'], width, label='Full Image Model')
            plt.bar(x + width / 2, family_df['Segment Model Accuracy'], width, label='Segment Model')

            # Add significance markers
            for i, row in enumerate(family_df.itertuples()):
                if row.Significant:
                    plt.plot(i, max(row._4, row._5) + 0.03, 'r*', markersize=10)

            plt.xlabel('Malware Family')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy by Malware Family (* indicates statistically significant difference)')
            plt.xticks(x, family_df['Family'], rotation=45, ha='right')
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'family_accuracy_comparison.png'), dpi=300)
            plt.close()

            # Save family analysis to CSV
            family_df.to_csv(os.path.join(self.output_dir, 'family_analysis.csv'), index=False)

    def run_analysis(self):
        """
        Run the complete analysis pipeline.

        Returns:
            Dictionary containing all comparison results
        """
        # 1. Load data
        if not self.load_data():
            print("ERROR: Could not load data. Aborting analysis.")
            return None

        # 2. Perform McNemar's test
        self.perform_mcnemars_test()
        print(
            f"McNemar's test result: statistic={self.comparison_results['mcnemar_statistic']:.4f}, p-value={self.comparison_results['mcnemar_p_value']:.6f}")
        print(f"Exact binomial test p-value: {self.comparison_results['exact_p_value']:.6f}")

        if self.comparison_results['is_significant_005']:
            print(f"The difference is statistically significant (p < 0.05)")
            print(f"Better model: {self.comparison_results['better_model']}")
        else:
            print("The difference is not statistically significant (p >= 0.05)")

        # 3. Calculate model metrics
        metrics = self.calculate_model_metrics()
        print("\nPerformance Metrics:")
        print(
            f"Full Image Model: Accuracy={metrics['full_model']['accuracy']:.4f}, F1={metrics['full_model']['f1']:.4f}")
        print(
            f"Segment Model: Accuracy={metrics['segment_model']['accuracy']:.4f}, F1={metrics['segment_model']['f1']:.4f}")
        print(
            f"Difference (Segment - Full): Accuracy={metrics['difference']['accuracy']:.4f}, F1={metrics['difference']['f1']:.4f}")

        # 4. Analyze disagreements
        self.analyze_disagreements()

        # 5. Analyze by family
        self.analyze_by_family()

        # 6. Create visualizations
        self.create_visualizations()

        # 7. Save results to JSON
        # Convert NumPy values to Python native types for JSON serialization
        def numpy_to_native(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: numpy_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_native(i) for i in obj]
            else:
                return obj

        # Convert all values to native Python types
        serializable_results = numpy_to_native(self.comparison_results)

        # Save to JSON file
        results_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nResults saved to {results_path}")

        return self.comparison_results


def parse_args():
    parser = argparse.ArgumentParser(description='Compare malware classification models using McNemar\'s test')
    parser.add_argument('--full_model_dir', type=str, required=True,
                        help='Directory containing full-image model results')
    parser.add_argument('--segment_model_dir', type=str, required=True,
                        help='Directory containing segment-based model results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save comparison results (defaults to segment_model_dir/comparison_results)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize comparator
    comparator = ModelComparator(
        full_model_dir=args.full_model_dir,
        segment_model_dir=args.segment_model_dir,
        output_dir=args.output_dir
    )

    # Run analysis
    results = comparator.run_analysis()

    if results:
        print("\nAnalysis completed successfully!")

        # Summarize findings
        print("\nSummary of findings:")
        if results['is_significant_005']:
            print(
                f"- There is a STATISTICALLY SIGNIFICANT difference between the models (p={results['mcnemar_p_value']:.6f})")
            if results['better_model'] == 'full_image':
                print("- The FULL IMAGE model performs better")
            else:
                print("- The SEGMENT-BASED model performs better")
        else:
            print(
                f"- There is NO statistically significant difference between the models (p={results['mcnemar_p_value']:.6f})")

        print(f"- Full Image Model Accuracy: {results['metrics']['full_model']['accuracy']:.4f}")
        print(f"- Segment Model Accuracy: {results['metrics']['segment_model']['accuracy']:.4f}")

        # If family analysis was performed, show top findings
        if 'family_analysis' in results:
            print("\nKey findings by malware family:")
            for family, data in results['family_analysis'].items():
                if data['is_significant']:
                    better = "FULL IMAGE" if data['better_model'] == 'full_image' else "SEGMENT"
                    print(f"- {family}: {better} model significantly better (p={data['mcnemar_p_value']:.4f})")

    return 0


if __name__ == "__main__":
    main()