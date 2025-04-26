#!/usr/bin/env python3
"""
McNemar Family Comparison Script

This script analyzes the per-family performance differences between the
full-image and segment-based models using the results from McNemar's test.
It provides detailed statistics and visualizations for each malware family.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze per-family model performance differences')
    parser.add_argument('--results_dir', type=str, default='results/segment_model/comparison_results',
                        help='Directory containing comparison results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results (defaults to results_dir)')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load comparison results
    results_path = os.path.join(args.results_dir, 'comparison_results.json')
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return 1

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Check if family analysis is available
    if 'family_analysis' not in results:
        print("Error: Family analysis data not found in results file")
        return 1

    family_analysis = results['family_analysis']

    # Convert to DataFrame for easier analysis
    family_data = []
    for family, data in family_analysis.items():
        family_data.append({
            'family': family,
            'sample_count': data['sample_count'],
            'full_model_accuracy': data['full_model_accuracy'],
            'segment_model_accuracy': data['segment_model_accuracy'],
            'accuracy_diff': data['accuracy_diff'],
            'mcnemar_p_value': data['mcnemar_p_value'],
            'is_significant': data['is_significant'],
            'better_model': data['better_model'],
            'b_full_correct_seg_wrong': data.get('b', None),  # ✅ disagreement 1
            'c_full_wrong_seg_correct': data.get('c', None)  # ✅ disagreement 2
        })

    # Create DataFrame
    df = pd.DataFrame(family_data)

    # Sort by absolute accuracy difference
    df['abs_diff'] = df['accuracy_diff'].abs()
    df_sorted = df.sort_values('abs_diff', ascending=False).reset_index(drop=True)

    # Add a column for difference direction
    df_sorted['diff_direction'] = df_sorted['accuracy_diff'].apply(
        lambda x: 'Full Model Better' if x < 0 else 'Segment Model Better' if x > 0 else 'Equal'
    )

    # Add significance marker
    df_sorted['significance'] = df_sorted['is_significant'].apply(
        lambda x: '* Significant' if x else 'Not Significant'
    )

    # Print overall summary
    print("\n=== FAMILY-WISE PERFORMANCE ANALYSIS ===\n")
    print(f"Total families analyzed: {len(df_sorted)}")

    sig_families = df_sorted[df_sorted['is_significant']]
    print(f"Families with statistically significant differences: {len(sig_families)}")

    full_better = df_sorted[df_sorted['accuracy_diff'] < 0]
    segment_better = df_sorted[df_sorted['accuracy_diff'] > 0]
    print(f"Families where full model performs better: {len(full_better)}")
    print(f"Families where segment model performs better: {len(segment_better)}")

    # Print detailed family results
    print("\n=== DETAILED FAMILY RESULTS (SORTED BY ACCURACY DIFFERENCE) ===\n")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    # Format DataFrame for display
    display_df = df_sorted[['family', 'sample_count', 'full_model_accuracy',
                            'segment_model_accuracy', 'accuracy_diff',
                            'mcnemar_p_value', 'is_significant', 'better_model']]
    display_df = display_df.rename(columns={
        'family': 'Family',
        'sample_count': 'Samples',
        'full_model_accuracy': 'Full Model Acc',
        'segment_model_accuracy': 'Segment Model Acc',
        'accuracy_diff': 'Acc Diff (Seg-Full)',
        'mcnemar_p_value': 'p-value',
        'is_significant': 'Significant',
        'better_model': 'Better Model'
    })

    # Format floating point columns
    for col in ['Full Model Acc', 'Segment Model Acc', 'Acc Diff (Seg-Full)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    display_df['p-value'] = display_df['p-value'].apply(lambda x: f"{x:.6f}")

    print(display_df.to_string(index=False))

    # Save to CSV
    csv_path = os.path.join(output_dir, 'mcnemar_family_comparison.csv')
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nSaved detailed family analysis to {csv_path}")

    # Create visualizations

    # 1. Bar chart of accuracy by family
    plt.figure(figsize=(14, 10))

    # Sort by family name for this chart
    df_name_sorted = df.sort_values('family')

    x = np.arange(len(df_name_sorted))
    width = 0.35

    plt.bar(x - width / 2, df_name_sorted['full_model_accuracy'],
            width, label='Full Image Model', color='blue', alpha=0.7)
    plt.bar(x + width / 2, df_name_sorted['segment_model_accuracy'],
            width, label='Segment Model', color='orange', alpha=0.7)

    # Add significance markers
    for i, row in enumerate(df_name_sorted.itertuples()):
        if row.is_significant:
            plt.plot(i, max(row.full_model_accuracy, row.segment_model_accuracy) + 0.02,
                     'r*', markersize=10)

    plt.xlabel('Malware Family')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Malware Family (* indicates statistically significant difference)')
    plt.xticks(x, df_name_sorted['family'], rotation=45, ha='right')
    plt.ylim(0.8, 1.05)  # Adjust as needed
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'mcnemar_family_accuracy.png'), dpi=300)

    # 2. Accuracy difference chart (sorted)
    plt.figure(figsize=(14, 10))

    # Use df_sorted (sorted by absolute difference)
    colors = ['red' if x < 0 else 'green' for x in df_sorted['accuracy_diff']]

    bars = plt.bar(df_sorted['family'], df_sorted['accuracy_diff'], color=colors, alpha=0.7)

    # Add significance markers
    for i, row in enumerate(df_sorted.itertuples()):
        if row.is_significant:
            plt.plot(i, row.accuracy_diff + (0.01 if row.accuracy_diff > 0 else -0.01),
                     'k*', markersize=10)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Malware Family')
    plt.ylabel('Accuracy Difference (Segment - Full)')
    plt.title('Accuracy Difference by Malware Family (* indicates statistical significance)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Add a legend for the significance marker
    plt.plot([], [], 'k*', markersize=10, label='Statistically Significant')
    plt.legend()

    plt.savefig(os.path.join(output_dir, 'mcnemar_family_difference.png'), dpi=300)

    # 3. Scatter plot: sample count vs accuracy difference
    plt.figure(figsize=(12, 8))

    plt.scatter(df['sample_count'], df['accuracy_diff'],
                c=['red' if sig else 'blue' for sig in df['is_significant']],
                alpha=0.7, s=100)

    # Add family labels to points
    for i, row in enumerate(df.itertuples()):
        plt.annotate(row.family,
                     (row.sample_count, row.accuracy_diff),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy Difference (Segment - Full)')
    plt.title('Accuracy Difference vs. Sample Count by Family')
    plt.grid(True, alpha=0.3)

    # Add a legend
    plt.scatter([], [], c='red', alpha=0.7, s=100, label='Statistically Significant')
    plt.scatter([], [], c='blue', alpha=0.7, s=100, label='Not Significant')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mcnemar_sample_vs_difference.png'), dpi=300)

    # 4. Heatmap of p-values
    plt.figure(figsize=(12, 8))

    # Create a new DataFrame with family and p-value columns
    heatmap_data = df_sorted[['family', 'mcnemar_p_value']].copy()

    # Apply -log10 transformation to p-values for better visualization
    # Add small constant to avoid log(0)
    heatmap_data['neg_log_p'] = -np.log10(heatmap_data['mcnemar_p_value'] + 1e-10)

    # Create a pivot table for the heatmap
    heatmap_pivot = pd.DataFrame({
        'family': heatmap_data['family'],
        'neg_log_p': heatmap_data['neg_log_p']
    }).set_index('family')

    # Plot heatmap
    sns.heatmap(heatmap_pivot.T, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('-log10(p-value) by Malware Family (higher values = more significant)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mcnemar_pvalue_heatmap.png'), dpi=300)

    print(f"\nAnalysis complete! Visualizations saved to {output_dir}")
    return 0


if __name__ == "__main__":
    main()