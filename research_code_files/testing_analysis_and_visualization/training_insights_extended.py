#!/usr/bin/env python3
"""
Script to generate advanced training insights visualizations and statistical analysis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from scipy import stats


def plot_metrics(history_df, output_dir):
    """
    Plot training metrics from history DataFrame

    Args:
        history_df: DataFrame containing training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    # 1. Accuracy and Loss curves
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['accuracy'], 'b-', linewidth=2, label='Training')
    plt.plot(history_df['val_accuracy'], 'r-', linewidth=2, label='Validation')
    plt.title('Model Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history_df['loss'], 'b-', linewidth=2, label='Training')
    plt.plot(history_df['val_loss'], 'r-', linewidth=2, label='Validation')
    plt.title('Model Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_loss_curves.png'), dpi=300, bbox_inches='tight')

    # 2. Precision, Recall, and AUC curves
    metrics = ['precision', 'recall', 'auc']
    if all(m in history_df.columns and f'val_{m}' in history_df.columns for m in metrics):
        plt.figure(figsize=(18, 6))

        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i + 1)
            plt.plot(history_df[metric], 'b-', linewidth=2, label=f'Training {metric.capitalize()}')
            plt.plot(history_df[f'val_{metric}'], 'r-', linewidth=2, label=f'Validation {metric.capitalize()}')
            plt.title(f'Model {metric.capitalize()}', fontsize=14)
            plt.ylabel(metric.capitalize(), fontsize=12)
            plt.xlabel('Epoch', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')

    # 3. Train vs Validation Gap Analysis
    plt.figure(figsize=(10, 6))

    acc_gap = history_df['accuracy'] - history_df['val_accuracy']
    loss_gap = history_df['loss'] - history_df['val_loss']

    plt.plot(acc_gap, 'g-', linewidth=2, label='Accuracy Gap (Train - Val)')
    plt.plot(loss_gap, 'm-', linewidth=2, label='Loss Gap (Train - Val)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title('Overfitting Analysis', fontsize=14)
    plt.ylabel('Gap', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')

    # 4. Learning Rate Analysis (if available)
    if 'lr' in history_df.columns:
        plt.figure(figsize=(10, 6))

        plt.plot(history_df['lr'], 'c-', linewidth=2)
        plt.title('Learning Rate Over Time', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')

    # 5. Combined Performance Dashboard
    plt.figure(figsize=(16, 12))

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history_df['accuracy'], 'b-', linewidth=2, label='Training')
    plt.plot(history_df['val_accuracy'], 'r-', linewidth=2, label='Validation')
    plt.title('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Loss
    plt.subplot(2, 2, 2)
    plt.plot(history_df['loss'], 'b-', linewidth=2, label='Training')
    plt.plot(history_df['val_loss'], 'r-', linewidth=2, label='Validation')
    plt.title('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Precision-Recall
    if 'precision' in history_df.columns and 'recall' in history_df.columns:
        plt.subplot(2, 2, 3)
        plt.plot(history_df['precision'], 'g-', linewidth=2, label='Precision')
        plt.plot(history_df['recall'], 'm-', linewidth=2, label='Recall')
        plt.title('Precision vs Recall (Training)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # Validation Precision-Recall
        plt.subplot(2, 2, 4)
        plt.plot(history_df['val_precision'], 'g-', linewidth=2, label='Precision')
        plt.plot(history_df['val_recall'], 'm-', linewidth=2, label='Recall')
        plt.title('Precision vs Recall (Validation)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')

    # 6. NEW: Moving Averages
    plt.figure(figsize=(14, 6))

    window_size = min(5, len(history_df))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['val_accuracy'], 'r-', alpha=0.3, label='Raw Val Accuracy')
    plt.plot(history_df['val_accuracy'].rolling(window=window_size).mean(), 'r-', linewidth=2,
             label=f'{window_size}-Epoch Moving Avg')
    plt.title('Val Accuracy Smoothed', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(history_df['val_loss'], 'b-', alpha=0.3, label='Raw Val Loss')
    plt.plot(history_df['val_loss'].rolling(window=window_size).mean(), 'b-', linewidth=2,
             label=f'{window_size}-Epoch Moving Avg')
    plt.title('Val Loss Smoothed', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moving_averages.png'), dpi=300, bbox_inches='tight')

    # 7. NEW: Correlation Heatmap
    plt.figure(figsize=(12, 10))

    # Select relevant columns for correlation
    metric_columns = [col for col in history_df.columns if any(
        m in col for m in ['accuracy', 'loss', 'precision', 'recall', 'auc'])]

    if len(metric_columns) >= 2:
        corr_matrix = history_df[metric_columns].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    square=True, linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Metric Correlation Heatmap', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')

    # 8. NEW: Performance Variability
    plt.figure(figsize=(12, 6))

    metrics_to_analyze = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
    epoch_groups = [
        (0, len(history_df) // 3),  # First third
        (len(history_df) // 3, 2 * len(history_df) // 3),  # Middle third
        (2 * len(history_df) // 3, len(history_df))  # Last third
    ]
    labels = ['First Third', 'Middle Third', 'Last Third']

    # Calculate variability for each metric in each epoch group
    variabilities = []
    for metric in metrics_to_analyze:
        if metric in history_df.columns:
            metric_vars = []
            for start, end in epoch_groups:
                metric_vars.append(history_df[metric].iloc[start:end].std())
            variabilities.append((metric, metric_vars))

    # Create grouped bar chart
    x = np.arange(len(labels))
    width = 0.2
    offsets = np.linspace(-0.3, 0.3, len(variabilities))

    for i, (metric, vars) in enumerate(variabilities):
        plt.bar(x + offsets[i], vars, width, label=metric)

    plt.xlabel('Training Progress', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.title('Metric Variability Throughout Training', fontsize=14)
    plt.xticks(x, labels)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_variability.png'), dpi=300, bbox_inches='tight')

    # Close all figures
    plt.close('all')


def analyze_training(history_df):
    """
    Analyze training patterns and provide insights

    Args:
        history_df: DataFrame containing training history

    Returns:
        Dictionary of insights
    """
    insights = {}

    # Number of epochs actually trained
    insights['total_epochs'] = len(history_df)

    # Find epoch with best validation accuracy
    best_val_acc_epoch = history_df['val_accuracy'].idxmax()
    insights['best_val_acc_epoch'] = best_val_acc_epoch
    insights['best_val_acc'] = history_df.loc[best_val_acc_epoch, 'val_accuracy']

    # Find epoch with best validation loss
    best_val_loss_epoch = history_df['val_loss'].idxmin()
    insights['best_val_loss_epoch'] = best_val_loss_epoch
    insights['best_val_loss'] = history_df.loc[best_val_loss_epoch, 'val_loss']

    # Calculate if model was overfitting
    if insights['total_epochs'] >= 5:
        # Check last 5 epochs for signs of overfitting
        last_5 = history_df.iloc[-5:]
        avg_acc_gap = (last_5['accuracy'] - last_5['val_accuracy']).mean()
        avg_loss_gap = (last_5['loss'] - last_5['val_loss']).mean()

        insights['avg_acc_gap_last_5'] = avg_acc_gap
        insights['avg_loss_gap_last_5'] = avg_loss_gap

        if avg_acc_gap > 0.05 and avg_loss_gap < 0:  # Typical signs of overfitting
            insights['overfitting'] = True
        else:
            insights['overfitting'] = False

    # Check if training plateaued
    if insights['total_epochs'] >= 10:
        # Check if validation accuracy improved in last 10 epochs
        last_10 = history_df.iloc[-10:]
        val_acc_change = last_10['val_accuracy'].iloc[-1] - last_10['val_accuracy'].iloc[0]

        insights['val_acc_change_last_10'] = val_acc_change

        if abs(val_acc_change) < 0.01:  # Less than 1% improvement
            insights['plateaued'] = True
        else:
            insights['plateaued'] = False

    return insights


def calculate_statistics(history_df):
    """
    Calculate advanced statistics from training history

    Args:
        history_df: DataFrame containing training history

    Returns:
        Dictionary of statistics
    """
    stats_results = {}

    # 1. Performance Variability Analysis
    stats_results['variability'] = {}
    for column in history_df.columns:
        stats_results['variability'][column] = {
            'mean': history_df[column].mean(),
            'median': history_df[column].median(),
            'std': history_df[column].std(),
            'min': history_df[column].min(),
            'max': history_df[column].max()
        }

    # 2. Correlation Analysis
    stats_results['correlations'] = {}

    # Precision-Recall Correlation (if available)
    if 'precision' in history_df.columns and 'recall' in history_df.columns:
        stats_results['correlations']['precision_recall'] = history_df['precision'].corr(history_df['recall'])
        stats_results['correlations']['val_precision_recall'] = history_df['val_precision'].corr(
            history_df['val_recall'])

    # Training-Validation Gaps correlation with epoch number
    if len(history_df) > 5:
        acc_gap = history_df['accuracy'] - history_df['val_accuracy']
        loss_gap = history_df['loss'] - history_df['val_loss']

        # Create a Series for epoch numbers with same index as acc_gap
        epoch_series = pd.Series(range(len(history_df)), index=history_df.index)

        stats_results['correlations']['acc_gap_vs_epoch'] = acc_gap.corr(epoch_series)
        stats_results['correlations']['loss_gap_vs_epoch'] = loss_gap.corr(epoch_series)

    # 3. Moving Averages and Trends
    window_size = min(5, len(history_df))
    if len(history_df) >= window_size:
        stats_results['moving_averages'] = {}

        # Calculate last moving average for key metrics
        for column in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
            if column in history_df.columns:
                moving_avg = history_df[column].rolling(window=window_size).mean()
                stats_results['moving_averages'][f'{column}_last_{window_size}'] = moving_avg.iloc[-1]

        # Calculate improvement rates (if enough data points)
        if len(history_df) >= 10:
            stats_results['improvement_rates'] = {}

            for column in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
                if column in history_df.columns:
                    # First half improvement rate
                    first_half = history_df[column].iloc[:len(history_df) // 2]
                    if not first_half.empty and len(first_half) > 1:
                        first_half_rate = (first_half.iloc[-1] - first_half.iloc[0]) / (len(first_half) - 1)
                        stats_results['improvement_rates'][f'{column}_first_half'] = first_half_rate

                    # Second half improvement rate
                    second_half = history_df[column].iloc[len(history_df) // 2:]
                    if not second_half.empty and len(second_half) > 1:
                        second_half_rate = (second_half.iloc[-1] - second_half.iloc[0]) / (len(second_half) - 1)
                        stats_results['improvement_rates'][f'{column}_second_half'] = second_half_rate

    # 4. Optimization Plateau Detection
    if len(history_df) >= 10:
        stats_results['plateau_metrics'] = {}

        # Calculate std of last 10 epochs for key metrics
        last_10 = history_df.iloc[-10:]
        for column in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
            if column in history_df.columns:
                std_last_10 = last_10[column].std()
                stats_results['plateau_metrics'][f'{column}_std_last_10'] = std_last_10

    # 5. Learning Rate Impact (if available)
    if 'lr' in history_df.columns and len(history_df) > 1:
        stats_results['learning_rate_impact'] = {}

        # Detect significant learning rate changes
        lr_changes = []
        for i in range(1, len(history_df)):
            if history_df['lr'].iloc[i] != history_df['lr'].iloc[i - 1]:
                lr_changes.append(i)

        # Calculate performance changes after learning rate changes
        if lr_changes:
            stats_results['learning_rate_impact']['change_epochs'] = lr_changes

            for change_epoch in lr_changes:
                if change_epoch < len(history_df) - 1:  # Ensure there's at least one epoch after change
                    window = min(3, len(history_df) - change_epoch - 1)  # Look at up to 3 epochs after change

                    for column in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
                        if column in history_df.columns:
                            before_change = history_df[column].iloc[change_epoch - 1]
                            after_changes = history_df[column].iloc[change_epoch:change_epoch + window]
                            avg_after_change = after_changes.mean()

                            key = f'{column}_change_after_lr_epoch_{change_epoch}'
                            stats_results['learning_rate_impact'][key] = avg_after_change - before_change

    return stats_results


def main():
    parser = argparse.ArgumentParser(description='Generate training insights visualizations')
    parser.add_argument('--history-file', type=str, required=True,
                        help='Path to training_history.csv file')
    parser.add_argument('--output-dir', type=str, default='training_insights',
                        help='Directory to save visualizations')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training history
    history_df = pd.read_csv(args.history_file)
    print(f"Loaded training history with {len(history_df)} epochs and {len(history_df.columns)} metrics")

    # Analyze training
    insights = analyze_training(history_df)

    # Calculate advanced statistics
    statistics = calculate_statistics(history_df)

    # Write results to a text file
    with open(os.path.join(args.output_dir, 'training_analysis.txt'), 'w') as f:
        f.write("=== MALWARE CLASSIFIER TRAINING ANALYSIS ===\n\n")

        # Basic Training Insights
        f.write("== BASIC TRAINING INSIGHTS ==\n")
        f.write(f"Total epochs trained: {insights['total_epochs']}\n")
        f.write(f"Best validation accuracy: {insights['best_val_acc']:.4f} (epoch {insights['best_val_acc_epoch']})\n")
        f.write(f"Best validation loss: {insights['best_val_loss']:.4f} (epoch {insights['best_val_loss_epoch']})\n\n")

        if 'overfitting' in insights:
            if insights['overfitting']:
                f.write("Overfitting detected:\n")
                f.write(f"- Average accuracy gap (train-val) in last 5 epochs: {insights['avg_acc_gap_last_5']:.4f}\n")
                f.write(f"- Average loss gap (train-val) in last 5 epochs: {insights['avg_loss_gap_last_5']:.4f}\n\n")
            else:
                f.write("No significant overfitting detected in the last 5 epochs\n\n")

        if 'plateaued' in insights:
            if insights['plateaued']:
                f.write("Training plateaued:\n")
                f.write(f"- Validation accuracy change in last 10 epochs: {insights['val_acc_change_last_10']:.4f}\n\n")
            else:
                f.write("Training was still improving when stopped\n\n")

        # Performance Variability
        f.write("== PERFORMANCE VARIABILITY ==\n")
        for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            if metric in statistics['variability']:
                f.write(f"{metric}:\n")
                m = statistics['variability'][metric]
                f.write(f"  Mean: {m['mean']:.4f}, Median: {m['median']:.4f}, Std: {m['std']:.4f}\n")
                f.write(f"  Range: [{m['min']:.4f}, {m['max']:.4f}]\n")

                # Calculate coefficient of variation for non-loss metrics
                if 'loss' not in metric and m['mean'] != 0:
                    cv = m['std'] / m['mean']
                    f.write(f"  Coefficient of Variation: {cv:.4f}\n")
                f.write("\n")

        # Correlation Analysis
        f.write("== CORRELATION ANALYSIS ==\n")
        if 'correlations' in statistics:
            for corr_name, corr_val in statistics['correlations'].items():
                f.write(f"{corr_name}: {corr_val:.4f}\n")

                # Add interpretation
                if 'gap_vs_epoch' in corr_name:
                    if corr_val > 0.7:
                        f.write("  Interpretation: Strong increase in gap over time (worsening overfitting)\n")
                    elif corr_val > 0.3:
                        f.write("  Interpretation: Moderate increase in gap over time (potential overfitting)\n")
                    elif corr_val < -0.7:
                        f.write("  Interpretation: Strong decrease in gap over time (improving generalization)\n")
                    elif corr_val < -0.3:
                        f.write(
                            "  Interpretation: Moderate decrease in gap over time (somewhat improving generalization)\n")
                    else:
                        f.write("  Interpretation: No clear trend in gap over time\n")
                elif 'precision_recall' in corr_name:
                    if corr_val > 0.7:
                        f.write("  Interpretation: Precision and recall improve together (good balance)\n")
                    elif corr_val < -0.7:
                        f.write(
                            "  Interpretation: Strong precision-recall tradeoff (model prioritizes one over the other)\n")
                    else:
                        f.write("  Interpretation: No strong relationship between precision and recall\n")
                f.write("\n")

        # Moving Averages and Trends
        if 'moving_averages' in statistics:
            f.write("== MOVING AVERAGES AND TRENDS ==\n")
            for ma_name, ma_val in statistics['moving_averages'].items():
                f.write(f"{ma_name}: {ma_val:.4f}\n")
            f.write("\n")

            if 'improvement_rates' in statistics:
                f.write("== IMPROVEMENT RATES ==\n")
                for rate_name, rate_val in statistics['improvement_rates'].items():
                    f.write(f"{rate_name}: {rate_val:.6f} per epoch\n")

                    # Compare first half vs second half
                    metric = rate_name.split('_first_half')[0] if '_first_half' in rate_name else \
                    rate_name.split('_second_half')[0]
                    other_half = f"{metric}_second_half" if '_first_half' in rate_name else f"{metric}_first_half"

                    if other_half in statistics['improvement_rates']:
                        other_val = statistics['improvement_rates'][other_half]
                        ratio = abs(rate_val / other_val) if other_val != 0 else float('inf')

                        if '_first_half' in rate_name and abs(rate_val) > abs(other_val) * 2:
                            f.write(f"  Note: Learning was {ratio:.1f}x faster in first half than second half\n")
                        elif '_second_half' in rate_name and abs(rate_val) > abs(other_val) * 2:
                            f.write(f"  Note: Learning was {ratio:.1f}x faster in second half than first half\n")
                f.write("\n")

        # Plateau Detection
        if 'plateau_metrics' in statistics:
            f.write("== PLATEAU DETECTION ==\n")
            for plateau_name, plateau_val in statistics['plateau_metrics'].items():
                f.write(f"{plateau_name}: {plateau_val:.6f}\n")

                # Add interpretation
                if plateau_val < 0.01 and ('accuracy' in plateau_name or 'loss' in plateau_name):
                    f.write("  Interpretation: Training has plateaued (very low variability in final epochs)\n")
                elif plateau_val < 0.03 and ('accuracy' in plateau_name or 'loss' in plateau_name):
                    f.write("  Interpretation: Training is approaching plateau (low variability in final epochs)\n")
            f.write("\n")

        # Learning Rate Impact
        if 'learning_rate_impact' in statistics:
            f.write("== LEARNING RATE IMPACT ==\n")
            if 'change_epochs' in statistics['learning_rate_impact']:
                f.write(
                    f"Learning rate changes occurred at epochs: {statistics['learning_rate_impact']['change_epochs']}\n\n")

                for key, val in statistics['learning_rate_impact'].items():
                    if key != 'change_epochs':
                        f.write(f"{key}: {val:.4f}\n")

                        # Add interpretation
                        if '_accuracy' in key and val > 0.01:
                            f.write("  Interpretation: Positive impact from learning rate change\n")
                        elif '_accuracy' in key and val < -0.01:
                            f.write("  Interpretation: Negative impact from learning rate change\n")
                        elif '_loss' in key and val < -0.01:
                            f.write("  Interpretation: Positive impact from learning rate change\n")
                        elif '_loss' in key and val > 0.01:
                            f.write("  Interpretation: Negative impact from learning rate change\n")
            else:
                f.write("No learning rate changes detected\n")
            f.write("\n")

        # Overall Recommendations
        f.write("== OVERALL RECOMMENDATIONS ==\n")

        if 'overfitting' in insights and insights['overfitting']:
            f.write("1. Use the model weights from epoch " + str(
                insights['best_val_acc_epoch']) + " (best validation performance)\n")
            f.write("2. Consider adding more regularization (dropout, L2) to combat overfitting\n")
            f.write("3. Implement data augmentation to improve generalization\n")
        elif 'plateaued' in insights and insights['plateaued']:
            f.write("1. Training reached a plateau - consider a different model architecture or features\n")
            f.write("2. Try a more aggressive learning rate schedule to escape plateau\n")
        else:
            f.write("1. Training was progressing well - consider training for more epochs\n")
            f.write("2. The model shows good generalization ability\n")

        # Print file path
        print(f"Analysis results saved to {os.path.join(args.output_dir, 'training_analysis.txt')}")

    # Plot metrics
    print(f"Generating visualizations in {args.output_dir}...")
    plot_metrics(history_df, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()