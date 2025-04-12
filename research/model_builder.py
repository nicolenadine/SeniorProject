#!/usr/bin/env python3
"""
Model Builder Module for Malware Classification System
Handles model architecture, compilation, and callbacks setup
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Reshape, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np


class CastLayer(Layer):
    """Custom Keras Layer to replace direct tf.cast() calls."""

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)


class ModelBuilder:
    def __init__(self, img_size=256, model_type='resnet18', channels=1):
        """
        Initialize the model builder.

        Args:
            img_size: Size of the input images (img_size x img_size).
            model_type: Type of model architecture to use.
            channels: Number of input channels (1 for grayscale, 3 for RGB).
        """
        self.img_size = img_size
        self.model_type = model_type
        self.channels = channels
        self.model = None
        self.conv_layers = []

    def build_model(self):
        """
        Build the CNN model based on the specified architecture.
        Returns:
            Built model and the list of convolutional layer names.
        """

        def focal_loss(gamma=2.0, alpha=0.05):  # Using alpha=0.5 for balanced classes
            def focal_loss_fixed(y_true, y_pred):
                # Clip to prevent numerical instability
                epsilon = K.epsilon()
                y_pred = K.clip(y_pred, epsilon, 1 - epsilon)

                # Calculate focal loss term
                cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
                loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy * y_true + \
                       (1 - alpha) * K.pow(y_pred, gamma) * cross_entropy * (1 - y_true)

                return K.mean(loss)

            return focal_loss_fixed

        def se_block(x, reduction_ratio=16):
            # Store input
            input_channels = x.shape[-1]

            # Squeeze operation (global average pooling)
            se = GlobalAveragePooling2D()(x)
            se = Reshape((1, 1, input_channels))(se)

            # Excitation operation (two FC layers)
            se = Dense(input_channels // reduction_ratio, activation='relu')(se)
            se = Dense(input_channels, activation='sigmoid')(se)

            # Scale the input
            x = Multiply()([x, se])
            return x

        print(f"Building {self.model_type} model...")

        if self.model_type == 'resnet18':
            # ResNet18-like architecture
            inputs = Input(shape=(self.img_size, self.img_size, self.channels))

            # Initial convolution block
            x = Conv2D(32, kernel_size=7, strides=2, padding='same', kernel_regularizer=l2(0.001))(inputs)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

            # Reduced filter sizes
            filter_sizes = [24, 48, 96, 192]  # Down from [32, 64, 128, 256]

            for filters in filter_sizes:
                # Single residual block instead of two
                residual = x
                x = Conv2D(filters, kernel_size=3, padding='same', kernel_regularizer=l2(0.001))(x)
                x = BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
                x = Conv2D(filters, kernel_size=3, padding='same', kernel_regularizer=l2(0.001))(x)
                x = BatchNormalization()(x)

                # Reshape residual if needed
                if residual.shape[-1] != filters:
                    residual = Conv2D(filters, kernel_size=1, padding='same', kernel_regularizer=l2(0.001))(residual)
                    residual = BatchNormalization()(residual)

                # Add residual and apply SE block
                x = tf.keras.layers.add([x, residual])
                x = tf.keras.layers.Activation('relu')(x)
                x = se_block(x)
                x = Dropout(0.2)(x)

                # Downsampling
                if filters != filter_sizes[-1]:
                    x = Conv2D(filters * 2, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2(0.001))(x)
                    x = BatchNormalization()(x)
                    x = tf.keras.layers.Activation('relu')(x)
                    x = se_block(x)

            # Global pooling and classification head
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.4)(x)
            # Single output neuron with sigmoid activation for binary classification
            outputs = Dense(1, activation='sigmoid')(x)

            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer=Adam(learning_rate=0.0002),
                loss=focal_loss(alpha=0.5, gamma=2.0),
                metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.AUC(name='auc')]
            )

            # Store the convolutional layer names for Grad-CAM analysis
            self.conv_layers = [layer.name for layer in self.model.layers
                                if isinstance(layer, tf.keras.layers.Conv2D)]

        elif self.model_type == 'simple_cnn': # used during testing but not used in final models
            # Simple CNN architecture
            inputs = Input(shape=(self.img_size, self.img_size, self.channels))

            # First convolutional block
            x = Conv2D(32, kernel_size=3, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = MaxPooling2D(pool_size=2)(x)
            x = Dropout(0.3)(x)

            # Second convolutional block
            x = Conv2D(64, kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = MaxPooling2D(pool_size=2)(x)
            x = Dropout(0.3)(x)

            # Third convolutional block
            x = Conv2D(128, kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = MaxPooling2D(pool_size=2)(x)
            x = Dropout(0.3)(x)

            # Classification head
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)

            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer=Adam(learning_rate=0.0003),
                loss='binary_crossentropy',
                metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.AUC(name='auc')]
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Print model summary
        self.model.summary()

        return self.model, self.conv_layers

    def setup_callbacks(self, output_dir):
        """
        Setup callbacks for model training.

        Args:
            output_dir: Directory to save model checkpoints and logs.

        Returns:
            List of callbacks for training.
        """
        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Model checkpoint
        checkpoint = ModelCheckpoint(
            os.path.join(output_dir, 'model', 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)

        # Custom detailed progress callback
        class DetailedProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"\nEpoch {epoch + 1}/{self.params['epochs']} completed")
                print(f"Training metrics: Loss={logs['loss']:.4f}, Accuracy={logs['accuracy']:.4f}")
                print(f"Validation metrics: Loss={logs['val_loss']:.4f}, Accuracy={logs['val_accuracy']:.4f}")
                print(f"Current learning rate: {tf.keras.backend.get_value(self.model.optimizer.learning_rate):.8f}")

        callbacks.append(DetailedProgress())

        return callbacks


    def save_model_summary(self, output_dir):
        """
        Save model architecture summary to a file.

        Args:
            output_dir: Directory to save the model summary.
        """
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return

        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)

        model_summary_file = os.path.join(output_dir, 'model', 'architecture_summary.txt')
        with open(model_summary_file, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        print(f"Model summary saved to {model_summary_file}")

    def save_model(self, output_dir, filename='final_model.h5'):
        """
        Save the trained model.

        Args:
            output_dir: Directory to save the model.
            filename: Name of the model file.
        """
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return

        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)

        model_path = os.path.join(output_dir, 'model', filename)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """
        Load a previously trained model.

        Args:
            model_path: Path to the model file.

        Returns:
            Loaded model and list of convolutional layer names.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")

            # Get convolutional layer names for Grad-CAM analysis
            self.conv_layers = [layer.name for layer in self.model.layers
                                if isinstance(layer, tf.keras.layers.Conv2D)]

            return self.model, self.conv_layers
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, []


def train(args, data_handler, model_builder, output_dir, fold_index=0):
    """
    Train a new model

    Args:
        args: Command-line arguments
        data_handler: Initialized DataHandler object
        output_dir: Directory to save results
        fold_index: Index of the fold to use for training (default: 0)

    Returns:
        Trained model and data handler
    """
    print("=== Starting Training Task ===")

    # Check if we have precomputed folds from cross-validation
    if hasattr(args, 'precomputed_folds') and args.precomputed_folds:
        # Use the precomputed data
        files = args.precomputed_folds['files']
        labels = args.precomputed_folds['labels']
        folds = args.precomputed_folds['folds']
        print("Using precomputed folds from cross-validation")
    else:
        # Otherwise load and split the data from scratch
        print("No precomputed folds found, creating new folds")
        files, labels = data_handler.load_and_balance_data(args.data_dir, malware_target=8500)
        n_splits = args.n_splits if hasattr(args, 'n_splits') else 5
        folds = data_handler.create_stratified_folds(files, labels, n_splits=n_splits)

    # Validate fold index
    if fold_index >= len(folds):
        print(f"Warning: Requested fold {fold_index} but only {len(folds)} folds available.")
        fold_index = 0

    # Use the specified fold for training
    fold = folds[fold_index]

    # Directly use the pre-split indices from data_handler instead of doing our own split
    train_files = [files[idx] for idx in fold['train_idx']]
    train_labels = [labels[idx] for idx in fold['train_idx']]
    val_files = [files[idx] for idx in fold['val_idx']]
    val_labels = [labels[idx] for idx in fold['val_idx']]
    test_files = [files[idx] for idx in fold['test_idx']]
    test_labels = [labels[idx] for idx in fold['test_idx']]

    # Update data_handler with these splits
    data_handler.train_files = train_files
    data_handler.train_labels = train_labels
    data_handler.val_files = val_files
    data_handler.val_labels = val_labels
    data_handler.test_files = test_files
    data_handler.test_labels = test_labels

    # Copy over family labels if available in precomputed data
    if hasattr(args, 'precomputed_folds') and args.precomputed_folds and 'family_labels' in args.precomputed_folds and args.precomputed_folds['family_labels'] is not None:
        data_handler.family_labels = args.precomputed_folds['family_labels']

    # Record dataset statistics for logging and config saving
    data_stats = {
        'train_samples': len(train_files),
        'val_samples': len(val_files),
        'test_samples': len(test_files),
        'num_folds': len(folds),
        'current_fold': fold_index
    }

    # Save configuration and dataset statistics
    import json
    config = vars(args).copy()
    # Remove precomputed_folds from config to avoid saving potentially large data
    if 'precomputed_folds' in config:
        del config['precomputed_folds']
    config.update(data_stats)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

    # Build the model if not already built
    if model_builder.model is None:
        model = ModelBuilder()
        model_builder.model, model_builder.conv_layers = model.build_model()

    # Save the model summary to the specified output directory
    model_builder.save_model_summary(output_dir)

    # Set up training callbacks
    callbacks = model_builder.setup_callbacks(output_dir)

    # Initialize trainer with the updated data handler
    from trainer import Trainer
    trainer = Trainer(
        model=model_builder.model,
        data_handler=data_handler,
        output_dir=output_dir,
        epochs=args.epochs
    )

    # Train model
    history = trainer.train(callbacks=callbacks)

    # Save training history
    import pandas as pd
    history_df = pd.DataFrame(history.history)
    history_csv_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    # Save the trained model
    model_builder.save_model(output_dir)

    print(f"Training completed. Results saved to {output_dir}")
    return model_builder.model, data_handler