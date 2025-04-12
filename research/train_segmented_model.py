#!/usr/bin/env python3
"""
Segmented Model Training with Ensemble Voting
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
import cv2
from datetime import datetime
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from tensorflow.keras.models import load_model

# Import your existing modules
from data_handler import DataHandler
from model_builder import ModelBuilder, CastLayer
from evaluator import Evaluator
# The visualization module will be used for GradCAM analysis
from visualization import GradCAMGenerator


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmented models with ensemble voting')

    # Data directories
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing malware and benign samples')
    parser.add_argument('--results_dir', type=str, default='results/segmented_model',
                        help='Directory to save results')
    parser.add_argument('--full_model_results', type=str, default='results/full_image_model',
                        help='Directory with full model results (for comparison)')

    # Model parameters
    parser.add_argument('--img_size', type=int, default=256,
                        help='Size of the full input images')
    parser.add_argument('--segment_size', type=int, default=128,
                        help='Size of the image segments')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['resnet18', 'simple_cnn'],
                        help='Type of model architecture to use')

    # Segmentation parameters
    parser.add_argument('--num_segments', type=int, default=6,
                        help='Number of segments to divide the image into')
    parser.add_argument('--segment_overlap', type=float, default=0.2,
                        help='Overlap between segments (0.0-0.5)')
    parser.add_argument('--voting_threshold', type=float, default=0.5,
                        help='Threshold for ensemble voting')
    parser.add_argument('--voting_method', type=str, default='majority',
                        choices=['majority', 'average', 'weighted'],
                        help='Method for combining segment predictions')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--malware_target', type=int, default=8500,
                        help='Target number of malware samples to balance data')

    # Analysis parameters
    parser.add_argument('--collect_gradcam', action='store_true',
                        help='Generate GradCAM visualizations')

    return parser.parse_args()


class SegmentGenerator:
    """Handler for creating and managing image segments"""

    def __init__(self, full_size, segment_size, num_segments=6, overlap=0.2):
        """
        Initialize the segment generator

        Args:
            full_size: Size of the full image (assumed square)
            segment_size: Size of the segments (assumed square)
            num_segments: Number of segments to divide the image into
            overlap: Percentage of overlap between segments (0.0-0.5)
        """
        self.full_size = full_size
        self.segment_size = segment_size
        self.num_segments = num_segments
        self.overlap = overlap

        # Calculate segment locations based on number of segments and overlap
        self.segment_locations = self._calculate_segment_locations()

    #Generalized approach for segment locations
    def _calculate_segment_locations(self):
        locations = []
        rows = int(np.sqrt(self.num_segments))
        cols = self.num_segments // rows + (1 if self.num_segments % rows else 0)

        step_y = self.full_size // rows
        step_x = self.full_size // cols
        margin_y = int(step_y * self.overlap)
        margin_x = int(step_x * self.overlap)

        for i in range(rows):
            for j in range(cols):
                if len(locations) < self.num_segments:
                    locations.append((j * (step_x - margin_x), i * (step_y - margin_y)))

        return locations

    def create_segments(self, image):
        """
        Create segments from a single image

        Args:
            image: The full image to segment (expected shape: [height, width, channels])

        Returns:
            List of segment images, each with shape [segment_size, segment_size, channels]
        """
        segments = []

        # Extract segments from the full image
        for x, y in self.segment_locations:
            # Ensure we don't go outside the image bounds
            max_x = min(x + self.segment_size, self.full_size)
            max_y = min(y + self.segment_size, self.full_size)

            # Extract the segment - handle both gray and color images
            if len(image.shape) == 3:  # Color or grayscale with channel dimension
                segment = image[y:max_y, x:max_x, :]
            else:  # Grayscale without channel dimension
                segment = image[y:max_y, x:max_x]

            # Resize if the segment is not the expected size
            if segment.shape[0] != self.segment_size or segment.shape[1] != self.segment_size:
                if len(segment.shape) == 3:
                    segment = cv2.resize(segment, (self.segment_size, self.segment_size))
                else:
                    segment = cv2.resize(segment, (self.segment_size, self.segment_size))
                    segment = np.expand_dims(segment, axis=-1)  # Add channel dimension

            segments.append(segment)

        return segments

    def create_segment_dataset(self, images, labels):
        """
        Create a dataset of segments from a set of images

        Args:
            images: Array of full-size images
            labels: Corresponding labels

        Returns:
            Dictionary of segment datasets, each with format: {'images': [], 'labels': []}
        """
        segment_datasets = {}

        for i in range(self.num_segments):
            segment_datasets[i] = {'images': [], 'labels': []}

        # Process each full image
        for i, (image, label) in enumerate(zip(images, labels)):
            segments = self.create_segments(image)

            # Add each segment to its respective dataset
            for seg_idx, segment in enumerate(segments):
                if seg_idx < self.num_segments:  # Ensure we don't exceed the number of segments
                    segment_datasets[seg_idx]['images'].append(segment)
                    segment_datasets[seg_idx]['labels'].append(label)

        # Convert lists to numpy arrays
        for i in range(self.num_segments):
            segment_datasets[i]['images'] = np.array(segment_datasets[i]['images'])
            segment_datasets[i]['labels'] = np.array(segment_datasets[i]['labels'])

        return segment_datasets

    def visualize_segmentation(self, image, save_path=None):
        """
        Visualize how an image is segmented

        Args:
            image: The full image to visualize segmentation on
            save_path: Path to save the visualization
        """
        # Create a copy of the image for visualization
        if len(image.shape) == 3 and image.shape[2] == 1:
            # Convert single-channel to 3-channels for visualization
            viz_image = np.tile(image, (1, 1, 3))
        elif len(image.shape) == 2:
            # Add channel dimension and convert to 3-channels
            viz_image = np.stack([image] * 3, axis=-1)
        else:
            # Already multi-channel
            viz_image = image.copy()

        # Normalize to 0-255 if needed
        if viz_image.max() <= 1.0:
            viz_image = (viz_image * 255).astype(np.uint8)

        # Draw segment boundaries
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255)  # Light Blue
        ]

        for i, (x, y) in enumerate(self.segment_locations):
            color = colors[i % len(colors)]
            # Calculate segment boundaries
            max_x = min(x + self.segment_size, self.full_size)
            max_y = min(y + self.segment_size, self.full_size)

            # Draw rectangle
            cv2.rectangle(viz_image, (x, y), (max_x, max_y), color, 2)
            # Add segment number
            cv2.putText(viz_image, str(i + 1), (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display or save the visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(viz_image)
        plt.title('Image Segmentation Visualization')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class SegmentDataGenerator:
    def __init__(self, file_paths, labels, segment_generator, batch_size=32, is_training=False):
        self.file_paths = file_paths
        self.labels = labels
        self.segment_generator = segment_generator
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_segments = segment_generator.num_segments
        self.img_size = segment_generator.full_size

    def generate(self, segment_idx):
        """Generate batches of segment data for a specific segment index"""
        num_samples = len(self.file_paths)
        indices = np.arange(num_samples)

        if self.is_training:
            np.random.shuffle(indices)

        # Process in batches
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            # Initialize arrays for the batch
            batch_size = len(batch_indices)
            batch_segments = np.zeros((batch_size, self.segment_generator.segment_size,
                                       self.segment_generator.segment_size, 1))
            batch_labels = np.zeros(batch_size)

            # Load and process images in this batch
            for i, idx in enumerate(batch_indices):
                # Load single image
                file_path = self.file_paths[idx]
                try:
                    # Load grayscale image
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"Failed to load image: {file_path}")

                    # Resize to expected dimensions
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    img = img / 255.0  # Normalize

                    # Create segments and take the one at segment_idx
                    segments = self.segment_generator.create_segments(img)
                    if segment_idx < len(segments):
                        batch_segments[i] = segments[segment_idx]

                    # Store label
                    batch_labels[i] = self.labels[idx]
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
                    # Use zeros for this sample

            yield batch_segments, batch_labels



class SegmentedModelTrainer:
    """Handler for training and evaluating segmented models"""

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

        # Initialize segment generator
        self.segment_generator = SegmentGenerator(
            full_size=args.img_size,
            segment_size=args.segment_size,
            num_segments=args.num_segments,
            overlap=args.segment_overlap
        )

        # Will be populated during training
        self.segment_models = []
        self.ensemble_performance = {}
        self.segment_performance = []
        self.computational_metrics = {
            'training_time': 0,
            'inference_time': 0,
            'inference_time_per_sample': 0,
            'total_parameters': 0,
            'model_size_mb': 0,
            'flops_per_segment': 0,
            'total_flops': 0,
            'memory_usage_mb': 0
        }

    def setup_directories(self):
        """
        Set up directories for saving results
        """
        # Main results directory
        os.makedirs(self.args.results_dir, exist_ok=True)

        # Sub-directories for different types of results
        self.model_dir = os.path.join(self.args.results_dir, 'models')
        self.metrics_dir = os.path.join(self.args.results_dir, 'metrics')
        self.viz_dir = os.path.join(self.args.results_dir, 'visualizations')
        self.gradcam_dir = os.path.join(self.args.results_dir, 'gradcam')
        self.comparison_dir = os.path.join(self.args.results_dir, 'comparison')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.gradcam_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)

        # Segment-specific directories
        for i in range(self.args.num_segments):
            segment_dir = os.path.join(self.model_dir, f'segment_{i + 1}')
            os.makedirs(segment_dir, exist_ok=True)

    def prepare_data(self):
        """
        Load data using the same splits as the full model
        """
        print("Loading data using splits from full model...")

        # Check if splits are available
        splits_dir = os.path.join(self.args.full_model_results, 'data_splits')

        if not os.path.exists(splits_dir):
            print("Error: Data splits from full model not found. Please run full model training first.")
            raise FileNotFoundError(f"Data splits not found in {splits_dir}")

        # Load the splits
        with open(os.path.join(splits_dir, 'train_files.txt'), 'r') as f:
            train_files = f.read().splitlines()
        with open(os.path.join(splits_dir, 'train_labels.txt'), 'r') as f:
            train_labels = [int(x) for x in f.read().splitlines()]

        with open(os.path.join(splits_dir, 'val_files.txt'), 'r') as f:
            val_files = f.read().splitlines()
        with open(os.path.join(splits_dir, 'val_labels.txt'), 'r') as f:
            val_labels = [int(x) for x in f.read().splitlines()]

        with open(os.path.join(splits_dir, 'test_files.txt'), 'r') as f:
            test_files = f.read().splitlines()
        with open(os.path.join(splits_dir, 'test_labels.txt'), 'r') as f:
            test_labels = [int(x) for x in f.read().splitlines()]

        # Load family labels if available
        family_labels_path = os.path.join(splits_dir, 'family_labels.txt')
        if os.path.exists(family_labels_path):
            with open(family_labels_path, 'r') as f:
                self.data_handler.family_labels = f.read().splitlines()

        # Update the data handler
        self.data_handler.train_files = train_files
        self.data_handler.train_labels = train_labels
        self.data_handler.val_files = val_files
        self.data_handler.val_labels = val_labels
        self.data_handler.test_files = test_files
        self.data_handler.test_labels = test_labels

        # Set up TensorFlow datasets
        self.data_handler.setup_data_generators()

        # Load images for segmentation
        print("Loading images for segmentation...")

        # ... (rest of the code to load images into memory for segmentation) ...

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

        print(f"Data preparation complete with identical splits to full model. "
              f"Train: {split_info['train_samples']}, Val: {split_info['val_samples']}, "
              f"Test: {split_info['test_samples']} samples")

        return split_info

    def build_and_train_segment_model(self, segment_idx):
        """
        Build and train a model for a specific segment using generators

        Args:
            segment_idx: Index of the segment to train

        Returns:
            Trained model and training history
        """
        print(f"\nTraining model for segment {segment_idx + 1}/{self.args.num_segments}")

        # Create segment-specific directory
        segment_dir = os.path.join(self.model_dir, f'segment_{segment_idx + 1}')

        # Initialize model builder for this segment
        model_builder = ModelBuilder(
            img_size=self.args.segment_size,
            model_type=self.args.model_type,
            channels=1  # Assuming grayscale images
        )

        # Build model
        model, conv_layers = model_builder.build_model()

        # Setup callbacks
        callbacks = model_builder.setup_callbacks(segment_dir)

        # Create data generators
        train_generator = SegmentDataGenerator(
            self.data_handler.train_files,
            self.data_handler.train_labels,
            self.segment_generator,
            batch_size=self.args.batch_size,
            is_training=True
        )

        val_generator = SegmentDataGenerator(
            self.data_handler.val_files,
            self.data_handler.val_labels,
            self.segment_generator,
            batch_size=self.args.batch_size
        )

        # Calculate steps per epoch
        steps_per_epoch = len(self.data_handler.train_files) // self.args.batch_size
        validation_steps = len(self.data_handler.val_files) // self.args.batch_size

        # Ensure at least one step
        steps_per_epoch = max(1, steps_per_epoch)
        validation_steps = max(1, validation_steps)

        # Get class distribution from a small sample to determine class weights
        # We'll process a few batches to get a representative sample
        class_counts = {0: 0, 1: 0}
        sample_count = 0
        sample_size = min(500, len(self.data_handler.train_files))  # Sample up to 500 images

        for X_batch, y_batch in train_generator.generate(segment_idx):
            for label in y_batch:
                class_counts[int(label)] = class_counts.get(int(label), 0) + 1

            sample_count += len(y_batch)
            if sample_count >= sample_size:
                break

        # Compute class weights
        class_weight = None
        if len(class_counts) > 1 and class_counts[0] > 0 and class_counts[1] > 0:
            total_samples = sum(class_counts.values())
            weight_benign = (1 / class_counts[0]) * (total_samples / 2.0)
            weight_malware = (1 / class_counts[1]) * (total_samples / 2.0)
            class_weight = {0: weight_benign, 1: weight_malware}
            print(f"Using class weights: {class_weight}")

            # Create a wrapper model that applies class weights during training
            class WeightedModel(tf.keras.Model):
                def __init__(self, original_model, class_weights):
                    super(WeightedModel, self).__init__()
                    self.original_model = original_model
                    self.class_weights = class_weights
                    # Copy input/output specs
                    self._input_spec = self.original_model._input_spec

                def call(self, inputs, training=False):
                    return self.original_model(inputs, training=training)

                def train_step(self, data):
                    if len(data) == 3:
                        x, y, sample_weight = data
                    else:
                        x, y = data
                        sample_weight = None

                    # Apply class weights based on true labels
                    class_weights_tensor = tf.where(
                        tf.equal(tf.cast(y, tf.int32), 1),
                        tf.ones_like(y, dtype=tf.float32) * self.class_weights[1],
                        tf.ones_like(y, dtype=tf.float32) * self.class_weights[0]
                    )

                    # Combine with any existing sample weights
                    if sample_weight is not None:
                        sample_weight = sample_weight * class_weights_tensor
                    else:
                        sample_weight = class_weights_tensor

                    # Run a regular training step with the new sample weights
                    with tf.GradientTape() as tape:
                        y_pred = self(x, training=True)
                        loss = self.compiled_loss(
                            y, y_pred, sample_weight=sample_weight,
                            regularization_losses=self.losses
                        )

                    # Compute gradients
                    trainable_vars = self.trainable_variables
                    gradients = tape.gradient(loss, trainable_vars)

                    # Update weights
                    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                    # Update metrics
                    self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

                    # Return a dict mapping metric names to current value
                    results = {m.name: m.result() for m in self.metrics}
                    results.update({"loss": loss})
                    return results

                # Ensure the model has all the same properties as the original model
                @property
                def trainable_weights(self):
                    return self.original_model.trainable_weights

                @property
                def non_trainable_weights(self):
                    return self.original_model.non_trainable_weights

                @property
                def weights(self):
                    return self.original_model.weights

                @property
                def variables(self):
                    return self.original_model.variables

                @property
                def trainable_variables(self):
                    return self.original_model.trainable_variables

                @property
                def non_trainable_variables(self):
                    return self.original_model.non_trainable_variables

            # Wrap the original model
            weighted_model = WeightedModel(model, class_weight)

            # Re-compile the model with the same metrics as used in your ModelBuilder class
            weighted_model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.AUC(name='auc')]
            )

            # Use the weighted model instead
            model = weighted_model

        # Record training start time
        train_start = time.time()

        # Train the model using the generator (without class_weight parameter)
        print(f"Training segment {segment_idx + 1} model with generator")
        history = model.fit(
            train_generator.generate(segment_idx),
            steps_per_epoch=steps_per_epoch,
            epochs=self.args.epochs,
            validation_data=val_generator.generate(segment_idx),
            validation_steps=validation_steps,
            callbacks=callbacks,
            # Removed class_weight parameter
            verbose=1
        )

        # Record training end time
        train_end = time.time()
        training_time = train_end - train_start

        # Save final model
        model.save(os.path.join(segment_dir, 'final_model.h5'))

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(segment_dir, 'training_history.csv'), index=False)

        # Create test generator for evaluation
        test_generator = SegmentDataGenerator(
            self.data_handler.test_files,
            self.data_handler.test_labels,
            self.segment_generator,
            batch_size=self.args.batch_size
        )

        # Prepare test data and predictions
        y_true = []
        y_pred_prob = []
        test_steps = max(1, len(self.data_handler.test_files) // self.args.batch_size)

        # Collect predictions on test data
        for i, (X_batch, y_batch) in enumerate(test_generator.generate(segment_idx)):
            if i >= test_steps:
                break

            batch_preds = model.predict(X_batch, verbose=0)
            y_pred_prob.extend(batch_preds.flatten())
            y_true.extend(y_batch)

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_prob = np.array(y_pred_prob)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Evaluate model
        test_loss = model.evaluate(
            test_generator.generate(segment_idx),
            steps=test_steps,
            verbose=1
        )

        # Map metric names to values
        metric_names = ['loss', 'accuracy', 'precision', 'recall', 'auc']
        test_results = {metric_names[i]: test_loss[i] for i in range(min(len(test_loss), len(metric_names)))}

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Get class distribution for reporting
        class_distribution = {}
        for label in y_true:
            label_int = int(label)
            class_distribution[label_int] = class_distribution.get(label_int, 0) + 1

        # Save metrics
        segment_metrics = {
            'segment_idx': segment_idx,
            'training_time': training_time,
            'test_metrics': test_results,
            'confusion_matrix': cm.tolist(),
            'class_distribution': class_distribution
        }

        with open(os.path.join(segment_dir, 'metrics.json'), 'w') as f:
            json.dump(segment_metrics, f, indent=4)

        # Calculate model size
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32 (4 bytes)

        model_metrics = {
            'parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'model_size_mb': float(model_size_mb)
        }

        # Save model metrics
        with open(os.path.join(segment_dir, 'model_metrics.json'), 'w') as f:
            json.dump(model_metrics, f, indent=4)

        return {
            'model': model,
            'history': history,
            'metrics': segment_metrics,
            'model_metrics': model_metrics
        }

    def train_all_segment_models(self):
        """
        Train models for all segments
        """
        self.segment_results = []
        total_training_time = 0
        total_parameters = 0
        total_model_size = 0

        for segment_idx in range(self.args.num_segments):
            result = self.build_and_train_segment_model(segment_idx)
            self.segment_results.append(result)

            # Extract models for later use
            self.segment_models.append(result['model'])

            # Accumulate metrics
            total_training_time += result['metrics']['training_time']
            total_parameters += result['model_metrics']['parameters']
            total_model_size += result['model_metrics']['model_size_mb']

            # Add segment performance
            self.segment_performance.append({
                'segment': segment_idx + 1,
                'accuracy': result['metrics']['test_metrics']['accuracy'],
                'precision': result['metrics']['test_metrics']['precision'],
                'recall': result['metrics']['test_metrics']['recall'],
                'auc': result['metrics']['test_metrics']['auc']
            })

        # Update computational metrics
        self.computational_metrics.update({
            'training_time': total_training_time,
            'total_parameters': total_parameters,
            'model_size_mb': total_model_size,
            'avg_parameters_per_segment': total_parameters / self.args.num_segments,
            'avg_size_per_segment': total_model_size / self.args.num_segments
        })

        # Save segment performance
        segment_performance_df = pd.DataFrame(self.segment_performance)
        segment_performance_df.to_csv(os.path.join(self.metrics_dir, 'segment_performance.csv'), index=False)

        # Create segment performance visualization
        plt.figure(figsize=(12, 8))

        # Plot segment performance metrics
        metrics = ['accuracy', 'precision', 'recall', 'auc']
        for metric in metrics:
            values = [perf[metric] for perf in self.segment_performance]
            plt.plot(range(1, self.args.num_segments + 1), values, 'o-', label=metric)

        plt.xticks(range(1, self.args.num_segments + 1))
        plt.xlabel('Segment')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics by Segment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'segment_performance.png'), dpi=300)
        plt.close()

        return self.segment_results

    def ensemble_predict(self, test_files, test_labels):
        """
        Make predictions using the ensemble of segment models in a memory-efficient way

        Args:
            test_files: List of file paths to test images
            test_labels: List of labels for test images

        Returns:
            Ensemble predictions
        """
        all_segment_predictions = []

        # Record inference start time
        inference_start = time.time()

        # Process in batches to save memory
        batch_size = 32
        for start_idx in range(0, len(test_files), batch_size):
            end_idx = min(start_idx + batch_size, len(test_files))
            batch_files = test_files[start_idx:end_idx]

            batch_segment_preds = np.zeros((len(batch_files), self.args.num_segments))

            # Process each file in the batch
            for i, file_path in enumerate(batch_files):
                try:
                    # Load single image
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"Failed to load image: {file_path}")

                    # Resize and normalize
                    img = cv2.resize(img, (self.args.img_size, self.args.img_size))
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    img = img / 255.0  # Normalize

                    # Create segments
                    segments = self.segment_generator.create_segments(img)

                    # Get predictions from each segment model
                    for seg_idx, segment in enumerate(segments):
                        if seg_idx < len(self.segment_models):
                            # Add batch dimension for prediction
                            segment_batch = np.expand_dims(segment, axis=0)
                            pred = self.segment_models[seg_idx].predict(segment_batch, verbose=0)
                            batch_segment_preds[i, seg_idx] = float(pred[0][0])
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
                    # Use 0.5 (uncertain prediction) for this sample
                    batch_segment_preds[i, :] = 0.5

            # Add batch predictions to overall results
            all_segment_predictions.append(batch_segment_preds)

        # Combine batch results
        all_segment_predictions = np.vstack(all_segment_predictions)

        # Record inference end time and compute metrics as before
        inference_end = time.time()
        inference_time = inference_end - inference_start

        # Apply ensemble voting method (same as before)
        if self.args.voting_method == 'majority':
            # Convert probabilities to binary predictions using threshold
            binary_predictions = (all_segment_predictions > self.args.voting_threshold).astype(int)
            # Count votes for malware class
            ensemble_votes = np.sum(binary_predictions, axis=1)
            # Majority vote
            ensemble_predictions = (ensemble_votes > (self.args.num_segments / 2)).astype(int)
            # Calculate probabilities based on proportion of positive votes
            ensemble_probabilities = ensemble_votes / self.args.num_segments
        elif self.args.voting_method == 'average':
            # Average the probabilities from all segments
            ensemble_probabilities = np.mean(all_segment_predictions, axis=1)
            # Apply threshold to get final predictions
            ensemble_predictions = (ensemble_probabilities > self.args.voting_threshold).astype(int)
        elif self.args.voting_method == 'weighted':
            # Use weights based on validation performance
            weights = np.array([perf['auc'] for perf in self.segment_performance])
            weights = weights / np.sum(weights)  # Normalize

            # Apply weights
            ensemble_probabilities = np.zeros(len(test_files))
            for i, weight in enumerate(weights):
                ensemble_probabilities += all_segment_predictions[:, i] * weight

            ensemble_predictions = (ensemble_probabilities > self.args.voting_threshold).astype(int)

        return {
            'segment_predictions': all_segment_predictions,
            'ensemble_predictions': ensemble_predictions,
            'ensemble_probabilities': ensemble_probabilities,
            'inference_time': inference_time
        }

    def evaluate_ensemble(self):
        """
        Evaluate the ensemble performance on test data
        """
        print("\nEvaluating ensemble performance...")

        # Get ensemble predictions
        ensemble_results = self.ensemble_predict(self.X_test_full)
        ensemble_pred = ensemble_results['ensemble_predictions']
        ensemble_prob = ensemble_results['ensemble_probabilities']

        # Calculate metrics
        accuracy = np.mean(ensemble_pred == self.y_test)
        cm = confusion_matrix(self.y_test, ensemble_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(self.y_test, ensemble_prob)
        roc_auc = auc(fpr, tpr)

        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, ensemble_prob)
        avg_precision = average_precision_score(self.y_test, ensemble_prob)

        # Prepare ensemble metrics
        ensemble_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'voting_method': self.args.voting_method,
            'voting_threshold': self.args.voting_threshold
        }

        # Save ensemble metrics
        with open(os.path.join(self.metrics_dir, 'ensemble_metrics.json'), 'w') as f:
            json.dump(ensemble_metrics, f, indent=4)

        # Save ROC curve data
        roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        roc_data.to_csv(os.path.join(self.metrics_dir, 'ensemble_roc_data.csv'), index=False)

        # Save PR curve data
        pr_data = pd.DataFrame({'precision': precision_curve, 'recall': recall_curve})
        pr_data.to_csv(os.path.join(self.metrics_dir, 'ensemble_pr_data.csv'), index=False)

        # Save ensemble predictions
        np.save(os.path.join(self.metrics_dir, 'ensemble_predictions.npy'), ensemble_pred)
        np.save(os.path.join(self.metrics_dir, 'ensemble_probabilities.npy'), ensemble_prob)
        np.save(os.path.join(self.metrics_dir, 'segment_predictions.npy'),
                ensemble_results['segment_predictions'])

        # Visualize ensemble metrics
        self.visualize_ensemble_metrics(ensemble_metrics, fpr, tpr, precision_curve, recall_curve)

        # Save additional data needed for statistical analysis
        with open(os.path.join(self.metrics_dir, 'computational_metrics.json'), 'w') as f:
            json.dump(self.computational_metrics, f, indent=4)

        # Store ensemble performance for later use
        self.ensemble_performance = ensemble_metrics

        print(f"Ensemble evaluation complete. Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")

        return ensemble_metrics

    def evaluate_ensemble_from_files(self):
        """
        Evaluate the ensemble performance on test data using file paths
        """
        print("\nEvaluating ensemble performance...")

        # Get ensemble predictions directly from files
        ensemble_results = self.ensemble_predict(
            self.data_handler.test_files,
            self.data_handler.test_labels
        )

        ensemble_pred = ensemble_results['ensemble_predictions']
        ensemble_prob = ensemble_results['ensemble_probabilities']

        # Use test labels directly from data handler
        y_test = np.array(self.data_handler.test_labels)

        # Calculate metrics
        accuracy = np.mean(ensemble_pred == y_test)
        cm = confusion_matrix(y_test, ensemble_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, ensemble_prob)
        roc_auc = auc(fpr, tpr)

        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, ensemble_prob)
        avg_precision = average_precision_score(y_test, ensemble_prob)

        # Prepare ensemble metrics
        ensemble_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'voting_method': self.args.voting_method,
            'voting_threshold': self.args.voting_threshold
        }

        # Save ensemble metrics
        with open(os.path.join(self.metrics_dir, 'ensemble_metrics.json'), 'w') as f:
            json.dump(ensemble_metrics, f, indent=4)

        # Save ROC curve data
        roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        roc_data.to_csv(os.path.join(self.metrics_dir, 'ensemble_roc_data.csv'), index=False)

        # Save PR curve data
        pr_data = pd.DataFrame({'precision': precision_curve, 'recall': recall_curve})
        pr_data.to_csv(os.path.join(self.metrics_dir, 'ensemble_pr_data.csv'), index=False)

        # Save ensemble predictions
        np.save(os.path.join(self.metrics_dir, 'ensemble_predictions.npy'), ensemble_pred)
        np.save(os.path.join(self.metrics_dir, 'ensemble_probabilities.npy'), ensemble_prob)

        # If segment predictions are available, save them too
        if 'segment_predictions' in ensemble_results:
            np.save(os.path.join(self.metrics_dir, 'segment_predictions.npy'),
                    ensemble_results['segment_predictions'])

        # Visualize ensemble metrics
        self.visualize_ensemble_metrics(ensemble_metrics, fpr, tpr, precision_curve, recall_curve)

        # Save additional data needed for statistical analysis
        with open(os.path.join(self.metrics_dir, 'computational_metrics.json'), 'w') as f:
            json.dump(self.computational_metrics, f, indent=4)

        # Store ensemble performance for later use
        self.ensemble_performance = ensemble_metrics

        print(f"Ensemble evaluation complete. Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")

        return ensemble_metrics

    def save_feature_vectors(self):
        """
        Extract and save feature vectors from the penultimate layer of both models.
        This data will be used later for feature correlation analysis.
        """
        print("Extracting feature vectors for later analysis...")

        # Sample a subset of test data to keep memory usage reasonable
        sample_size = min(500, len(self.data_handler.test_files))
        sample_indices = np.random.choice(len(self.data_handler.test_files), sample_size, replace=False)

        # Sample files and labels
        sample_files = [self.data_handler.test_files[i] for i in sample_indices]
        sample_labels = [self.data_handler.test_labels[i] for i in sample_indices]

        # Record file paths so we can match features later
        feature_data = {
            "files": sample_files,
            "labels": sample_labels,
            "full_model_path": os.path.join(self.args.full_model_results, 'model', 'final_model.h5'),
            "segment_model_paths": [os.path.join(self.model_dir, f'segment_{i + 1}', 'final_model.h5')
                                    for i in range(self.args.num_segments)]
        }

        # Create directory for feature data
        feature_dir = os.path.join(self.args.results_dir, 'feature_data')
        os.makedirs(feature_dir, exist_ok=True)

        # Save metadata about the features we'll need to extract later
        with open(os.path.join(feature_dir, 'feature_extraction_info.json'), 'w') as f:
            json.dump(feature_data, f, indent=4)

        print(f"Feature extraction information saved to {feature_dir}")

    def save_error_concordance_data(self):
        """
        Save data needed for error concordance analysis between full model and ensemble.
        """
        # Load full model predictions
        full_model_pred_path = os.path.join(self.args.full_model_results, 'metrics', 'y_pred.npy')
        if os.path.exists(full_model_pred_path):
            full_model_preds = np.load(full_model_pred_path)

            # Load ensemble predictions - either from the dictionary or from saved file
            ensemble_preds = None
            if hasattr(self, 'ensemble_performance') and isinstance(self.ensemble_performance, dict):
                # Try different potential key names
                for key in ['ensemble_predictions', 'predictions', 'y_pred']:
                    if key in self.ensemble_performance:
                        ensemble_preds = self.ensemble_performance[key]
                        break

            # If not found in the dictionary, try loading from file
            if ensemble_preds is None:
                ensemble_pred_path = os.path.join(self.metrics_dir, 'ensemble_predictions.npy')
                if os.path.exists(ensemble_pred_path):
                    ensemble_preds = np.load(ensemble_pred_path)
                else:
                    print("Warning: Ensemble predictions not found. Skipping error concordance analysis.")
                    return

            # Get true labels
            y_test = np.array(self.data_handler.test_labels)

            # Create concordance data
            concordance_data = {
                "full_model_correct": (full_model_preds == y_test).astype(int),
                "ensemble_correct": (ensemble_preds == y_test).astype(int),
                "true_labels": y_test.tolist(),
                "test_files": self.data_handler.test_files
            }

            # Save data
            concordance_dir = os.path.join(self.comparison_dir, 'error_concordance')
            os.makedirs(concordance_dir, exist_ok=True)

            with open(os.path.join(concordance_dir, 'error_concordance_data.json'), 'w') as f:
                json.dump(concordance_data, f, indent=4)

            print(f"Error concordance data saved to {concordance_dir}")
        else:
            print(f"Full model predictions not found at {full_model_pred_path}. Skipping error concordance analysis.")


    def visualize_ensemble_metrics(self, metrics, fpr, tpr, precision_curve, recall_curve):
        """
        Create visualizations for ensemble metrics

        Args:
            metrics: Dictionary with ensemble metrics
            fpr, tpr: ROC curve data
            precision_curve, recall_curve: PR curve data
        """
        # Create confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malware'],
                    yticklabels=['Benign', 'Malware'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Ensemble Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'ensemble_confusion_matrix.png'), dpi=300)
        plt.close()

        # Create ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'ensemble_roc_curve.png'), dpi=300)
        plt.close()

        # Create PR curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, lw=2,
                 label=f'PR curve (AP = {metrics["avg_precision"]:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Ensemble Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'ensemble_pr_curve.png'), dpi=300)
        plt.close()

        # Create performance comparison bar chart
        plt.figure(figsize=(12, 8))

        # Get segment performance
        segment_metrics = {
            'accuracy': [perf['accuracy'] for perf in self.segment_performance],
            'precision': [perf['precision'] for perf in self.segment_performance],
            'recall': [perf['recall'] for perf in self.segment_performance],
            'auc': [perf['auc'] for perf in self.segment_performance]
        }

        # Add ensemble performance
        ensemble_values = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'auc': metrics['roc_auc']
        }

        # Plot each metric group
        metric_names = ['accuracy', 'precision', 'recall', 'auc']
        x = np.arange(len(metric_names))
        width = 0.1  # Width of bars

        # Plot segment bars
        for i in range(self.args.num_segments):
            segment_values = [segment_metrics[m][i] for m in metric_names]
            plt.bar(x + (i - self.args.num_segments / 2) * width, segment_values,
                    width, label=f'Segment {i + 1}')

        # Plot ensemble bar
        ensemble_values_list = [ensemble_values[m] for m in metric_names]
        plt.bar(x + (self.args.num_segments / 2) * width, ensemble_values_list,
                width, label='Ensemble', color='red')

        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Comparison: Segments vs Ensemble')
        plt.xticks(x, metric_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, 'performance_comparison.png'), dpi=300)
        plt.close()

    def compare_with_full_model(self):
        """
        Compare ensemble performance with full image model
        """
        print("\nComparing ensemble with full image model...")

        # Check if full model results directory exists
        if not os.path.exists(self.args.full_model_results):
            print(f"Full model results directory not found: {self.args.full_model_results}")
            return None

        # Load full model metrics
        full_metrics_path = os.path.join(self.args.full_model_results, 'metrics', 'classification_report.json')

        if not os.path.exists(full_metrics_path):
            print(f"Full model metrics not found: {full_metrics_path}")
            return None

        try:
            with open(full_metrics_path, 'r') as f:
                full_metrics = json.load(f)

            # Extract metrics for class 1 (malware)
            full_precision = full_metrics['1']['precision']
            full_recall = full_metrics['1']['recall']
            full_f1 = full_metrics['1']['f1-score']

            # Get overall accuracy
            full_accuracy = full_metrics['accuracy']

            # Load ROC data if available
            try:
                full_roc_path = os.path.join(self.args.full_model_results, 'metrics', 'roc_data.csv')
                full_roc_data = pd.read_csv(full_roc_path)
                full_auc = auc(full_roc_data['fpr'], full_roc_data['tpr'])
            except Exception as e:
                print(f"Error loading full model ROC data: {e}")
                full_auc = None

            # Load computational metrics if available
            try:
                full_comp_path = os.path.join(self.args.full_model_results, 'metrics', 'computational_metrics.json')
                with open(full_comp_path, 'r') as f:
                    full_comp_metrics = json.load(f)
            except Exception as e:
                print(f"Error loading full model computational metrics: {e}")
                full_comp_metrics = {
                    'training_time': None,
                    'inference_time': None,
                    'inference_time_per_sample': None,
                    'parameters_count': None,
                    'model_size_mb': None
                }

            # Create comparison
            comparison = {
                'model_type': {
                    'full_model': 'Single model on full image',
                    'ensemble': f'Ensemble of {self.args.num_segments} segment models'
                },
                'performance_metrics': {
                    'accuracy': {
                        'full_model': full_accuracy,
                        'ensemble': self.ensemble_performance['accuracy']
                    },
                    'precision': {
                        'full_model': full_precision,
                        'ensemble': self.ensemble_performance['precision']
                    },
                    'recall': {
                        'full_model': full_recall,
                        'ensemble': self.ensemble_performance['recall']
                    },
                    'f1_score': {
                        'full_model': full_f1,
                        'ensemble': self.ensemble_performance['f1_score']
                    },
                    'auc': {
                        'full_model': full_auc,
                        'ensemble': self.ensemble_performance['roc_auc']
                    }
                },
                'computational_metrics': {
                    'training_time': {
                        'full_model': full_comp_metrics.get('training_time'),
                        'ensemble': self.computational_metrics['training_time']
                    },
                    'inference_time_per_sample': {
                        'full_model': full_comp_metrics.get('inference_time_per_sample'),
                        'ensemble': self.computational_metrics['inference_time_per_sample']
                    },
                    'model_size_mb': {
                        'full_model': full_comp_metrics.get('model_size_mb'),
                        'ensemble': self.computational_metrics['model_size_mb']
                    },
                    'parameters_count': {
                        'full_model': full_comp_metrics.get('parameters_count'),
                        'ensemble': self.computational_metrics['total_parameters']
                    }
                }
            }

            # Calculate speedup and efficiency gains
            if full_comp_metrics.get('inference_time_per_sample') and self.computational_metrics[
                'inference_time_per_sample']:
                speedup = full_comp_metrics['inference_time_per_sample'] / self.computational_metrics[
                    'inference_time_per_sample']
                comparison['computational_metrics']['speedup'] = float(speedup)

            if full_comp_metrics.get('model_size_mb') and self.computational_metrics['model_size_mb']:
                size_reduction = 1 - (self.computational_metrics['model_size_mb'] / full_comp_metrics['model_size_mb'])
                comparison['computational_metrics']['size_reduction_percentage'] = float(size_reduction * 100)

            # Save comparison
            with open(os.path.join(self.comparison_dir, 'full_vs_ensemble_comparison.json'), 'w') as f:
                json.dump(comparison, f, indent=4)

            # Create comparison visualizations
            self.visualize_comparison(comparison)

            print("Comparison with full model completed")
            return comparison

        except Exception as e:
            print(f"Error comparing with full model: {e}")
            return None

    def visualize_comparison(self, comparison):
        """
        Create visualizations comparing full model and ensemble

        Args:
            comparison: Dictionary with comparison metrics
        """
        # Performance metrics comparison
        plt.figure(figsize=(12, 8))

        # Get performance metrics
        metrics = list(comparison['performance_metrics'].keys())
        full_values = [comparison['performance_metrics'][m]['full_model'] for m in metrics]
        ensemble_values = [comparison['performance_metrics'][m]['ensemble'] for m in metrics]

        # Plot grouped bar chart
        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width / 2, full_values, width, label='Full Model')
        plt.bar(x + width / 2, ensemble_values, width, label='Segment Ensemble')

        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Metrics: Full Model vs Ensemble')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.comparison_dir, 'performance_comparison.png'), dpi=300)
        plt.close()

        # Computational metrics comparison
        plt.figure(figsize=(12, 8))

        # Get computational metrics (excluding calculated ones)
        comp_metrics = ['training_time', 'inference_time_per_sample', 'model_size_mb', 'parameters_count']
        comp_labels = ['Training Time (s)', 'Inference Time per Sample (s)', 'Model Size (MB)', 'Parameters Count']

        # Create normalized values (full model = 1.0)
        normalized_values = []
        for metric in comp_metrics:
            full_val = comparison['computational_metrics'][metric]['full_model']
            ensemble_val = comparison['computational_metrics'][metric]['ensemble']

            if full_val and ensemble_val:  # Ensure both values exist
                normalized_values.append(ensemble_val / full_val)
            else:
                normalized_values.append(None)

        # Filter out None values
        valid_indices = [i for i, val in enumerate(normalized_values) if val is not None]
        valid_metrics = [comp_labels[i] for i in valid_indices]
        valid_values = [normalized_values[i] for i in valid_indices]

        if valid_values:  # Ensure we have some valid metrics
            # Plot horizontal bar chart
            plt.barh(valid_metrics, valid_values, color='skyblue')
            plt.axvline(x=1.0, color='red', linestyle='--', label='Full Model Baseline')

            plt.xlabel('Relative Value (Full Model = 1.0)')
            plt.title('Computational Efficiency: Ensemble vs Full Model')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.comparison_dir, 'computational_comparison.png'), dpi=300)
        plt.close()

    def save_segmented_gradcam_heatmaps(self):
        """
        Generate and save GradCAM heatmaps for the segmented model approach with stratified sampling.
        Captures heatmaps for individual segments and provides metadata for later reconstruction.
        """
        print("\nGenerating GradCAM heatmaps for segmented models...")

        # Create GradCAM directory if it doesn't exist
        os.makedirs(self.gradcam_dir, exist_ok=True)

        # Get test files and labels
        test_files = self.data_handler.test_files
        test_labels = self.data_handler.test_labels

        # Stratified sampling based on malware families or binary classification
        has_family_info = hasattr(self.data_handler, 'family_labels') and self.data_handler.family_labels is not None

        if has_family_info:
            # Get family labels for test samples
            test_family_indices = []
            for i, file_path in enumerate(test_files):
                for j, all_file in enumerate(self.data_handler.all_files):
                    if file_path == all_file:
                        test_family_indices.append(j)
                        break

            test_families = [self.data_handler.family_labels[i] for i in test_family_indices]

            # Get unique families
            unique_families = list(set(test_families))
            print(f"Found {len(unique_families)} unique malware families and benign samples")

            # Determine samples per family - at least 20 samples per family if possible
            samples_per_family = 30
            total_expected_samples = samples_per_family * len(unique_families)
            print(f"Planning to sample approximately {total_expected_samples} images total")

            # Select samples from each family
            selected_indices = []
            for family in unique_families:
                family_indices = [i for i, f in enumerate(test_families) if f == family]

                # Take min of available samples or desired samples per family
                num_samples = min(samples_per_family, len(family_indices))

                # Randomly select samples from this family
                if len(family_indices) > 0:
                    family_selected = np.random.choice(family_indices, num_samples, replace=False)
                    selected_indices.extend(family_selected)
                    print(f"Selected {num_samples} samples from family '{family}'")
        else:
            # No family info available, just ensure balanced benign/malware
            benign_indices = [i for i, label in enumerate(test_labels) if label == 0]
            malware_indices = [i for i, label in enumerate(test_labels) if label == 1]

            # Select 150 samples from each class (300 total)
            samples_per_class = 150

            benign_selected = np.random.choice(
                benign_indices,
                min(samples_per_class, len(benign_indices)),
                replace=False
            )

            malware_selected = np.random.choice(
                malware_indices,
                min(samples_per_class, len(malware_indices)),
                replace=False
            )

            selected_indices = list(benign_selected) + list(malware_selected)
            print(f"Selected {len(benign_selected)} benign and {len(malware_selected)} malware samples")

        # Convert selected indices to actual files and labels
        test_indices = np.array(selected_indices)
        test_files = [self.data_handler.test_files[i] for i in test_indices]
        test_labels = [self.data_handler.test_labels[i] for i in test_indices]

        # Create sample directory structure
        samples_dir = os.path.join(self.gradcam_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)

        # Create directory for each segment model
        for seg_idx in range(self.args.num_segments):
            os.makedirs(os.path.join(self.gradcam_dir, f'segment_{seg_idx + 1}'), exist_ok=True)

        # Store metadata about samples and segment locations
        sample_metadata = {
            "test_indices": test_indices.tolist(),
            "test_files": test_files,
            "test_labels": test_labels,
            "segment_info": {
                "num_segments": self.args.num_segments,
                "segment_size": self.args.segment_size,
                "full_size": self.args.img_size,
                "overlap": self.args.segment_overlap,
                "segment_locations": [
                    {"x": loc[0], "y": loc[1]}
                    for loc in self.segment_generator.segment_locations
                ]
            }
        }

        # Add family information to metadata if available
        if has_family_info:
            family_counts = {}
            selected_families = [test_families[i] for i in selected_indices]
            for family in selected_families:
                family_counts[family] = family_counts.get(family, 0) + 1

            sample_metadata["family_info"] = {
                "families": selected_families,
                "counts": family_counts
            }

            print("Sample distribution by family:")
            for family, count in family_counts.items():
                print(f"  {family}: {count} samples")

        # Save metadata
        with open(os.path.join(self.gradcam_dir, 'sample_metadata.json'), 'w') as f:
            json.dump(sample_metadata, f, indent=4)

        # Process each test image
        for sample_idx, file_path in enumerate(test_files):
            print(f"Processing sample {sample_idx + 1}/{len(test_files)}: {os.path.basename(file_path)}")

            try:
                # Load and preprocess the image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"  Warning: Could not load image {file_path}")
                    continue

                img = cv2.resize(img, (self.args.img_size, self.args.img_size))

                # Save the original image
                sample_dir = os.path.join(samples_dir, f'sample_{sample_idx:03d}')
                os.makedirs(sample_dir, exist_ok=True)
                cv2.imwrite(os.path.join(sample_dir, 'original.png'), img)

                # Create segments
                segments = self.segment_generator.create_segments(
                    np.expand_dims(img, axis=-1) / 255.0
                )

                # Generate GradCAM for each segment
                for seg_idx, segment in enumerate(segments):
                    if seg_idx >= self.args.num_segments:
                        break

                    # Get the corresponding model
                    model = self.segment_models[seg_idx]

                    # Add batch dimension for prediction
                    segment_batch = np.expand_dims(segment, axis=0)

                    # Make prediction
                    pred = model.predict(segment_batch, verbose=0)
                    pred_class = int(pred[0][0] > 0.5)
                    pred_prob = float(pred[0][0])

                    # Find the last convolutional layer for GradCAM
                    last_conv_layer = None
                    for layer in reversed(model.layers):
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            last_conv_layer = layer.name
                            break

                    if last_conv_layer is None:
                        print(f"  Warning: No convolutional layer found for segment {seg_idx + 1}")
                        continue

                    # Save segment image and metadata
                    seg_dir = os.path.join(sample_dir, f'segment_{seg_idx + 1}')
                    os.makedirs(seg_dir, exist_ok=True)

                    # Rescale segment for visualization (0-255)
                    segment_viz = (segment * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(seg_dir, 'segment.png'), segment_viz)

                    # Save prediction info
                    pred_info = {
                        "predicted_class": pred_class,
                        "probability": pred_prob,
                        "last_conv_layer": last_conv_layer,
                        "segment_index": seg_idx,
                        "location": {
                            "x": self.segment_generator.segment_locations[seg_idx][0],
                            "y": self.segment_generator.segment_locations[seg_idx][1]
                        }
                    }

                    with open(os.path.join(seg_dir, 'prediction_info.json'), 'w') as f:
                        json.dump(pred_info, f, indent=4)

                    # Note: Actual GradCAM computation is deferred
                    # We're just saving the necessary info for later analysis

            except Exception as e:
                print(f"  Error processing sample {file_path}: {e}")

        # Save instructions for GradCAM reconstruction
        reconstruction_instructions = {
            "overview": "To create full-image GradCAM heatmaps, individual segment GradCAMs need to be computed and mapped back to their original positions.",
            "algorithm": [
                "1. Load each segment from 'segment.png'",
                "2. Generate GradCAM heatmap for each segment using the corresponding model and last_conv_layer",
                "3. Place each segment's heatmap at its original position (from location data)",
                "4. For overlapping regions, use maximum values for clearest visualization",
                "5. Normalize the final heatmap to 0-1 range"
            ],
            "requirements": {
                "segment_models": [
                    os.path.join(self.model_dir, f'segment_{i + 1}', 'final_model.h5')
                    for i in range(self.args.num_segments)
                ],
                "full_model_for_comparison": os.path.join(
                    self.args.full_model_results, 'model', 'final_model.h5'
                ),
                "full_model_gradcam": os.path.join(
                    self.args.full_model_results, 'gradcam'
                )
            },
            "example_code": """
    # Example code for future GradCAM generation (pseudocode)
    def generate_gradcam(model, img, layer_name):
        # Create a model that maps input to both conv layer output and predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # Gradient of loss with respect to output feature map
        grads = tape.gradient(loss, conv_outputs)

        # Average gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight output feature map with gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    # For full reconstruction:
    def reconstruct_full_gradcam(segment_heatmaps, segment_locations, segment_size, full_size):
        # Create empty heatmap
        full_heatmap = np.zeros((full_size, full_size))

        # Place each segment heatmap at its position
        for heatmap, (x, y) in zip(segment_heatmaps, segment_locations):
            # Ensure we don't go out of bounds
            x_end = min(x + segment_size, full_size)
            y_end = min(y + segment_size, full_size)

            # Calculate the portion of the segment that fits
            h_segment = x_end - x
            w_segment = y_end - y

            # Update heatmap, using maximum for overlaps
            segment_resized = cv2.resize(heatmap, (h_segment, w_segment))
            full_heatmap[y:y_end, x:x_end] = np.maximum(
                full_heatmap[y:y_end, x:x_end], 
                segment_resized
            )

        # Normalize final heatmap
        if np.max(full_heatmap) > 0:
            full_heatmap = full_heatmap / np.max(full_heatmap)

        return full_heatmap
    """
        }

        # Save reconstruction instructions
        with open(os.path.join(self.gradcam_dir, 'gradcam_reconstruction_guide.json'), 'w') as f:
            json.dump(reconstruction_instructions, f, indent=4)

        print(f"GradCAM preparation completed. Data saved to {self.gradcam_dir}")
        print("Note: Actual GradCAM computation will need to be performed during analysis.")

        return {
            "samples_processed": len(test_files),
            "gradcam_dir": self.gradcam_dir
        }

    def run(self):
        """
        Run the complete training and evaluation pipeline for segmented models
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting segmented model pipeline at {timestamp}")

        # Save configuration
        config = vars(self.args)
        config['timestamp'] = timestamp
        with open(os.path.join(self.args.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # 1. Prepare data and create segments
        data_info = self.prepare_data()

        # 2. Train models for all segments
        segment_results = self.train_all_segment_models()

        # 3. Evaluate ensemble performance
        ensemble_metrics = self.evaluate_ensemble_from_files()

        # 4. Compare with full model if available
        comparison = self.compare_with_full_model()

        # 5. Generate GradCAM visualizations if requested
        # 5. Save additional data for future statistical analysis
        self.save_feature_vectors()
        self.save_error_concordance_data()
        self.save_segmented_gradcam_heatmaps()

        return {
            'data_info': data_info,
            'segment_models': self.segment_models,
            'ensemble_metrics': ensemble_metrics,
            'comparison': comparison
        }


def main():
    args = parse_args()

    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run the segmented model pipeline
    trainer = SegmentedModelTrainer(args)
    results = trainer.run()

    # Return success
    return 0


if __name__ == "__main__":
    main()