#!/usr/bin/env python3
"""
Full Image Model Training with Comprehensive Data Collection
For Malware Classification Project
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import train_test_split

# Import your existing modules
from data_handler import DataHandler
from model_builder import ModelBuilder, CastLayer
from evaluator import Evaluator
# The visualization module will be used for GradCAM analysis
from visualization import GradCAMGenerator, analyze_by_family


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train full image model with comprehensive data collection')

    # Data directories
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing malware and benign samples')
    parser.add_argument('--results_dir', type=str, default='results/full_image_model',
                        help='Directory to save results')

    # Model parameters
    parser.add_argument('--img_size', type=int, default=256,
                        help='Size of the input images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'simple_cnn'],
                        help='Type of model architecture to use')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--malware_target', type=int, default=8500,
                        help='Target number of malware samples to balance data')

    # Analysis and visualization parameters
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate results during training')
    parser.add_argument('--save_frequency', type=int, default=5,
                        help='Epoch frequency for saving intermediate results')
    parser.add_argument('--collect_gradcam', action='store_true',
                        help='Generate GradCAM visualizations')
    parser.add_argument('--layer_name', type=str, default=None,
                        help='Specific layer to use for GradCAM (default: use last conv layer)')

    return parser.parse_args()

class FullModelTrainer:
    def __init__(self, args):
        """
        Initialize the trainer with command line arguments
        """
        self.args = args
        self.setup_directories()

        # Initialize data handler
        self.data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Initialize model builder
        self.model_builder = ModelBuilder(
            img_size=args.img_size,
            model_type=args.model_type,
            channels=1  # Assuming grayscale images
        )

        # These will be populated during training
        self.model = None
        self.history = None
        self.evaluation_results = None
        self.visualization_results = None

        # For storing performance metrics across epochs
        self.metrics_history = {
            'epoch': [],
            'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_auc': [],
        }

        # For computational efficiency metrics
        self.computational_metrics = {
            'training_time': 0,
            'inference_time': 0,
            'inference_time_per_sample': 0,
            'model_size_mb': 0,
            'parameters_count': 0,
            'flops': 0
        }

    def setup_directories(self):
        """
        Set up directories for saving results
        """
        # Main results directory
        os.makedirs(self.args.results_dir, exist_ok=True)

        # Sub-directories for different types of results
        self.model_dir = os.path.join(self.args.results_dir, 'model')
        self.metrics_dir = os.path.join(self.args.results_dir, 'metrics')
        self.viz_dir = os.path.join(self.args.results_dir, 'visualizations')
        self.gradcam_dir = os.path.join(self.args.results_dir, 'gradcam')
        self.intermediate_dir = os.path.join(self.args.results_dir, 'intermediate')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.gradcam_dir, exist_ok=True)

        if self.args.save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)

    def prepare_data(self):
        """
        Load and preprocess data with a simple stratified 70/15/15 split
        """
        print("Loading and preprocessing data...")

        # Load balanced data (benign and malware)
        all_files, all_labels = self.data_handler.load_and_balance_data(
            self.args.data_dir,
            malware_target=self.args.malware_target
        )

        # Create the splits using stratification
        from sklearn.model_selection import train_test_split

        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            all_files, all_labels, test_size=0.15, random_state=42, stratify=all_labels
        )

        val_size = 0.15 / 0.85
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels, test_size=val_size,
            random_state=42, stratify=train_val_labels
        )

        # Update the data handler
        self.data_handler.train_files = train_files
        self.data_handler.train_labels = train_labels
        self.data_handler.val_files = val_files
        self.data_handler.val_labels = val_labels
        self.data_handler.test_files = test_files
        self.data_handler.test_labels = test_labels

        # Save the splits for the segmented model to use
        splits_dir = os.path.join(self.args.results_dir, 'data_splits')
        os.makedirs(splits_dir, exist_ok=True)

        # Save the file lists and labels
        with open(os.path.join(splits_dir, 'train_files.txt'), 'w') as f:
            f.write('\n'.join(train_files))
        with open(os.path.join(splits_dir, 'train_labels.txt'), 'w') as f:
            f.write('\n'.join(map(str, train_labels)))

        with open(os.path.join(splits_dir, 'val_files.txt'), 'w') as f:
            f.write('\n'.join(val_files))
        with open(os.path.join(splits_dir, 'val_labels.txt'), 'w') as f:
            f.write('\n'.join(map(str, val_labels)))

        with open(os.path.join(splits_dir, 'test_files.txt'), 'w') as f:
            f.write('\n'.join(test_files))
        with open(os.path.join(splits_dir, 'test_labels.txt'), 'w') as f:
            f.write('\n'.join(map(str, test_labels)))

        # Save family labels if available
        if hasattr(self.data_handler, 'family_labels') and self.data_handler.family_labels is not None:
            with open(os.path.join(splits_dir, 'family_labels.txt'), 'w') as f:
                f.write('\n'.join(self.data_handler.family_labels))

        # Set up TensorFlow datasets
        self.data_handler.setup_data_generators()

        # Save split info as before
        split_info = {
            'train_samples': len(train_files),
            'val_samples': len(val_files),
            'test_samples': len(test_files),
            'malware_train': train_labels.count(1),
            'benign_train': train_labels.count(0),
            'malware_val': val_labels.count(1),
            'benign_val': val_labels.count(0),
            'malware_test': test_labels.count(1),
            'benign_test': test_labels.count(0)
        }

        with open(os.path.join(self.metrics_dir, 'data_split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=4)

        print(f"Data preparation complete. Train: {split_info['train_samples']}, "
              f"Val: {split_info['val_samples']}, Test: {split_info['test_samples']} samples")

        return split_info

    def build_model(self):
        """
        Build and compile the model
        """
        print(f"Building {self.args.model_type} model...")

        # Build the model
        self.model, self.conv_layers = self.model_builder.build_model()

        # Save model summary to file
        self.model_builder.save_model_summary(self.args.results_dir)

        # Calculate model size
        self.calculate_model_metrics()

        return self.model

    def calculate_model_metrics(self):
        """
        Calculate and save model size, parameters, and FLOPs
        """
        if self.model is None:
            return

        # Count total parameters
        trainable_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params

        # Estimate model size (parameters * 4 bytes for float32)
        model_size_bytes = total_params * 4
        model_size_mb = model_size_bytes / (1024 * 1024)

        # Try to estimate FLOPs if available
        try:
            from tensorflow.keras.utils import plot_model

            # Create a concrete function from the model
            input_shape = (1, self.args.img_size, self.args.img_size, 1)  # Batch size 1
            concrete_func = tf.function(self.model).get_concrete_function(
                tf.TensorSpec(input_shape, tf.float32))

            # Calculate FLOPs
            forward_graph = concrete_func.graph
            forward_graph_def = forward_graph.as_graph_def()

            # Use the tf.compat.v1.profiler to estimate FLOPs
            flops = tf.compat.v1.profiler.profile(
                forward_graph,
                options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
            flops_per_image = flops.total_float_ops
        except Exception as e:
            print(f"Error estimating FLOPs: {e}")
            flops_per_image = 0

        # Save computational metrics
        self.computational_metrics.update({
            'parameters_count': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'model_size_mb': float(model_size_mb),
            'flops': int(flops_per_image)
        })

        # Save to file
        with open(os.path.join(self.metrics_dir, 'model_metrics.json'), 'w') as f:
            json.dump(self.computational_metrics, f, indent=4)

        print(f"Model has {total_params:,} parameters, size: {model_size_mb:.2f} MB")
        if flops_per_image > 0:
            print(f"Estimated FLOPs per inference: {flops_per_image:,}")

    def setup_callbacks(self):
        """
        Setup callbacks for model training by using the configuration from model_builder
        """
        # Use the callback setup from ModelBuilder which includes the DetailedProgress callback
        return self.model_builder.setup_callbacks(self.args.results_dir)

    def train(self):
        """
        Train the model
        """
        print("\n=== Starting model training ===")

        callbacks = self.setup_callbacks()

        # Record training start time
        training_start = time.time()

        # Train model - pass None for class_weights if you want the model to handle class weighting
        print("Training model...")

        # Using class weights to handle imbalance
        class_weight = None
        if hasattr(self.data_handler, 'class_weights') and self.data_handler.class_weights:
            class_weight = self.data_handler.class_weights
            print(f"Using class weights: {class_weight}")

        history = self.model.fit(
            self.data_handler.train_dataset,
            epochs=self.args.epochs,
            validation_data=self.data_handler.val_dataset,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        # Record training end time
        training_end = time.time()
        training_time = training_end - training_start

        # Save training time
        self.computational_metrics['training_time'] = training_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

        # Save final model
        final_model_path = os.path.join(self.model_dir, 'final_model.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.metrics_dir, 'training_history.csv'), index=False)

        # Save metrics history
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(os.path.join(self.metrics_dir, 'detailed_metrics_history.csv'), index=False)

        self.history = history
        return history

    def evaluate(self):
        """
        Evaluate the model on test data
        """
        print("\n=== Evaluating model on test data ===")

        # Load test data
        X_test, y_test = self.data_handler.load_test_data()

        # Record inference start time
        inference_start = time.time()

        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # Record inference end time
        inference_end = time.time()
        inference_time = inference_end - inference_start

        # Save inference time metrics
        self.computational_metrics['inference_time'] = inference_time
        self.computational_metrics['inference_time_per_sample'] = inference_time / len(X_test)

        # Save the test file paths for sample identification
        np.save(os.path.join(self.metrics_dir, 'test_file_paths.npy'), np.array(self.data_handler.test_files))

        # Create a CSV with file paths and predictions for easy matching
        import pandas as pd
        test_predictions_df = pd.DataFrame({
            'file_path': self.data_handler.test_files,
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_probability': y_pred_prob.flatten()  # Flatten in case of 2D array
        })
        test_predictions_df.to_csv(os.path.join(self.metrics_dir, 'test_predictions.csv'), index=False)

        with open(os.path.join(self.metrics_dir, 'computational_metrics.json'), 'w') as f:
            json.dump(self.computational_metrics, f, indent=4)

        print(f"Inference completed in {inference_time:.2f} seconds "
              f"({self.computational_metrics['inference_time_per_sample'] * 1000:.2f} ms per sample)")

        # Initialize evaluator and get metrics
        evaluator = Evaluator(self.model, self.args.results_dir)
        eval_results = evaluator.evaluate(X_test, y_test)

        # Save raw predictions for further analysis
        np.save(os.path.join(self.metrics_dir, 'y_true.npy'), y_test)
        np.save(os.path.join(self.metrics_dir, 'y_pred.npy'), y_pred)
        np.save(os.path.join(self.metrics_dir, 'y_pred_prob.npy'), y_pred_prob)

        # Calculate additional metrics that aren't in the standard evaluator

        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        average_precision = average_precision_score(y_test, y_pred_prob)

        # Save PR curve data
        pr_data = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'thresholds': np.append(thresholds, 1.0)  # Add 1.0 to match the length of precision/recall
        })
        pr_data.to_csv(os.path.join(self.metrics_dir, 'precision_recall_data.csv'), index=False)

        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {average_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'precision_recall_curve.png'), dpi=300)
        plt.close()

        # Save threshold analysis - metrics at different threshold values
        threshold_values = np.linspace(0.1, 0.9, 9)
        threshold_metrics = []

        for threshold in threshold_values:
            y_pred_at_threshold = (y_pred_prob > threshold).astype(int).flatten()

            # Create confusion matrix at this threshold
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_at_threshold).ravel()

            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (
                                                                                                precision_val + recall_val) > 0 else 0

            threshold_metrics.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision_val,
                'recall': recall_val,
                'f1_score': f1,
                'true_positive': tp,
                'false_positive': fp,
                'true_negative': tn,
                'false_negative': fn
            })

        # Save threshold metrics
        threshold_df = pd.DataFrame(threshold_metrics)
        threshold_df.to_csv(os.path.join(self.metrics_dir, 'threshold_metrics.csv'), index=False)

        # Plot threshold vs metrics
        plt.figure(figsize=(12, 8))
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            plt.plot(threshold_df['threshold'], threshold_df[metric], label=metric)
        plt.xlabel('Classification Threshold')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics vs. Classification Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'threshold_metrics.png'), dpi=300)
        plt.close()

        # Save evaluation results
        self.evaluation_results = eval_results
        return eval_results

    def generate_visualizations(self):
        """
        Generate comprehensive visualizations for the model
        """
        print("\n=== Generating visualizations ===")

        # Some visualizations like confusion matrix and ROC curve
        # are already done by the Evaluator class

        # Load test data
        X_test, y_test = self.data_handler.load_test_data()

        # Add training convergence plot
        if self.history:
            plt.figure(figsize=(12, 10))

            # Plot loss and accuracy
            plt.subplot(2, 1, 1)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy During Training')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'training_convergence.png'), dpi=300)
            plt.close()

        # If GradCAM is requested
        if self.args.collect_gradcam and hasattr(self.data_handler, 'family_labels'):
            print("Generating GradCAM visualizations...")

            # Create family-based GradCAM analysis
            try:
                gradcam_results = analyze_by_family(
                    self.args,
                    model=self.model,
                    X_test=X_test,
                    y_test=y_test,
                    data_handler=self.data_handler
                )

                self.visualization_results = {
                    'gradcam_analysis': True,
                    'family_analysis': True if hasattr(self.data_handler, 'family_labels') else False
                }
            except Exception as e:
                print(f"Error during GradCAM analysis: {e}")
                self.visualization_results = {
                    'gradcam_analysis': False,
                    'error': str(e)
                }

        return self.visualization_results

    def run(self):
        """
        Run the complete training and evaluation pipeline
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting full model pipeline at {timestamp}")

        # Save configuration
        config = vars(self.args)
        config['timestamp'] = timestamp
        with open(os.path.join(self.args.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # 1. Prepare data
        data_info = self.prepare_data()

        # 2. Build model
        self.build_model()

        # 3. Train model
        self.train()

        # 4. Evaluate model
        self.evaluate()

        # 5. Generate visualizations
        self.generate_visualizations()

        print(f"\nTraining and evaluation pipeline completed! Results saved to {self.args.results_dir}")
        print("You can now use these results for your statistical analysis and visualizations.")

        return {
            'data_info': data_info,
            'model': self.model,
            'history': self.history,
            'evaluation_results': self.evaluation_results,
            'computational_metrics': self.computational_metrics
        }


def main():
    args = parse_args()

    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run the full training pipeline
    trainer = FullModelTrainer(args)
    results = trainer.run()

    # Return success
    return 0


if __name__ == "__main__":
    main()