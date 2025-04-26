#!/usr/bin/env python3
"""
Evaluator Module for Malware Classification System
Handles model evaluation and performance metrics visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


class Evaluator:
    def __init__(self, model, output_dir):
        """
        Initialize the evaluator

        Args:
            model: The trained TensorFlow model
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.output_dir = output_dir

        # Create output directories
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set

        Args:
            X_test: Test images as a NumPy array
            y_test: Test labels as a NumPy array

        Returns:
            Dictionary containing evaluation metrics and predictions
        """
        print("Evaluating the model on test data...")

        # Evaluate overall metrics
        test_results = self.model.evaluate(X_test, y_test, verbose=1)

        metrics = ["loss", "accuracy"]
        if len(test_results) > 2:
            metrics.extend(["precision", "recall", "auc"])

        print("\nTest Metrics:")
        metrics_dict = {}
        for i, metric in enumerate(metrics):
            if i < len(test_results):
                metrics_dict[metric] = test_results[i]
                print(f"{metric}: {test_results[i]:.4f}")

        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save the confusion matrix as a CSV file
        pd.DataFrame(cm).to_csv(os.path.join(self.output_dir, 'metrics', 'confusion_matrix.csv'))

        # Save the classification report as a JSON file
        with open(os.path.join(self.output_dir, 'metrics', 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)

        # Plot and save the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malware'],
                    yticklabels=['Benign', 'Malware'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'confusion_matrix.png'), dpi=300)

        # Calculate and plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Save ROC data for future use
        roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        roc_data.to_csv(os.path.join(self.output_dir, 'metrics', 'roc_data.csv'), index=False)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'roc_curve.png'), dpi=300)

        # Close all plots to free memory
        plt.close('all')

        print(f"Evaluation complete. Results saved to {self.output_dir}/metrics/")

        # Return evaluation results
        results = {
            'metrics': metrics_dict,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': roc_auc,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob
            }
        }

        return results
