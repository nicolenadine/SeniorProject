#!/usr/bin/env python3
"""
Script to generate training insights visualizations from training history
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


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


def main():
    parser = argparse.ArgumentParser(description='Generate training insights visualizations')
    parser.add_argument('--history-file', type=str, required=True,
                        help='Path to training_history.csv file')
    parser.add_argument('--output-dir', type=str, default='training_insights',
                        help='Directory to save visualizations')
    args = parser.parse_args()

    # Load training history
    history_df = pd.read_csv(args.history_file)
    print(f"Loaded training history with {len(history_df)} epochs and {len(history_df.columns)} metrics")

    # Analyze training
    insights = analyze_training(history_df)

    # Print insights
    print("\n=== Training Insights ===")
    print(f"Total epochs trained: {insights['total_epochs']}")
    print(f"Best validation accuracy: {insights['best_val_acc']:.4f} (epoch {insights['best_val_acc_epoch']})")
    print(f"Best validation loss: {insights['best_val_loss']:.4f} (epoch {insights['best_val_loss_epoch']})")

    if 'overfitting' in insights:
        if insights['overfitting']:
            print("\nOverfitting detected:")
            print(f"- Average accuracy gap (train-val) in last 5 epochs: {insights['avg_acc_gap_last_5']:.4f}")
            print(f"- Average loss gap (train-val) in last 5 epochs: {insights['avg_loss_gap_last_5']:.4f}")
        else:
            print("\nNo significant overfitting detected in the last 5 epochs")

    if 'plateaued' in insights:
        if insights['plateaued']:
            print("\nTraining plateaued:")
            print(f"- Validation accuracy change in last 10 epochs: {insights['val_acc_change_last_10']:.4f}")
        else:
            print("\nTraining was still improving when stopped")

    # Plot metrics
    print(f"\nGenerating visualizations in {args.output_dir}...")
    plot_metrics(history_df, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()