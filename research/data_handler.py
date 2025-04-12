#!/usr/bin/env python3
"""
Data Handler Module for Malware Classification System
Handles data loading, preprocessing, and dataset creation
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
import pickle
import glob
import random
from sklearn.model_selection import StratifiedKFold
import time


class DataHandler:
    def __init__(self, data_dir, img_size=256, batch_size=64):
        """
        Initialize the data handler

        Args:
            data_dir: Directory containing 'malware' and 'benign' subdirectories
            img_size: Size of the input images (img_size x img_size)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.family_to_id = {}  # Map family names to unique IDs
        self.family_labels = None  # Store family labels for each sample

        # Will be populated during data preprocessing
        self.train_files = []
        self.train_labels = []
        self.val_files = []
        self.val_labels = []
        self.test_files = []
        self.test_labels = []
        self.class_weights = {}

        # Additional attributes for full dataset tracking
        self.all_files = []  # All files (train + val + test)
        self.all_labels = []  # All labels (train + val + test)
        self.family_labels = None  # Family labels for all samples
        self.file_to_family_map = {}  # Mapping from file paths to family labels

        # TensorFlow datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def find_files(self, directory):
        """ Recursively find all files in the directory. """
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files

    def load_and_preprocess_data(self, save_splits=True, splits_dir=None):
        """
        Load images from the data directory and prepare datasets

        Args:
            save_splits: Whether to save the train/val/test splits
            splits_dir: Directory to save the splits

        Returns:
            Dictionary with dataset statistics
        """
        print("Loading and preprocessing data...")
        malware_dir = os.path.join(self.data_dir, 'malware')
        benign_dir = os.path.join(self.data_dir, 'Benign')

        # Check if directories exist
        if not os.path.exists(malware_dir) or not os.path.exists(benign_dir):
            raise FileNotFoundError(f"Malware or benign subdirectories not found in {self.data_dir}")

        # Recursively get files
        malware_files = self.find_files(malware_dir)
        benign_files = self.find_files(benign_dir)

        print(f"Total malware samples: {len(malware_files)}")
        print(f"Total benign samples: {len(benign_files)}")

        # Split data: train(70%), val(15%), test(15%)
        train_malware, temp_malware = train_test_split(malware_files, test_size=0.3, random_state=42)
        val_malware, test_malware = train_test_split(temp_malware, test_size=0.5, random_state=42)

        train_benign, temp_benign = train_test_split(benign_files, test_size=0.3, random_state=42)
        val_benign, test_benign = train_test_split(temp_benign, test_size=0.5, random_state=42)

        # Create combined datasets with labels
        self.train_files = train_malware + train_benign
        self.train_labels = [1] * len(train_malware) + [0] * len(train_benign)

        self.val_files = val_malware + val_benign
        self.val_labels = [1] * len(val_malware) + [0] * len(val_benign)

        self.test_files = test_malware + test_benign
        self.test_labels = [1] * len(test_malware) + [0] * len(test_benign)

        # Shuffle the datasets
        train_indices = np.random.permutation(len(self.train_files))
        self.train_files = [self.train_files[i] for i in train_indices]
        self.train_labels = [self.train_labels[i] for i in train_indices]

        val_indices = np.random.permutation(len(self.val_files))
        self.val_files = [self.val_files[i] for i in val_indices]
        self.val_labels = [self.val_labels[i] for i in val_indices]

        test_indices = np.random.permutation(len(self.test_files))
        self.test_files = [self.test_files[i] for i in test_indices]
        self.test_labels = [self.test_labels[i] for i in test_indices]

        # Calculate class weights for balanced training
        total_samples = len(self.train_files)
        n_malware = self.train_labels.count(1)
        n_benign = self.train_labels.count(0)

        weight_malware = (1 / n_malware) * (total_samples / 2.0)
        weight_benign = (1 / n_benign) * (total_samples / 2.0)

        self.class_weights = {0: weight_benign, 1: weight_malware}

        # Print statistics
        print(f"Train set: {len(self.train_files)} samples")
        print(f"Validation set: {len(self.val_files)} samples")
        print(f"Test set: {len(self.test_files)} samples")
        print(f"Class weights: {self.class_weights}")

        # Create TensorFlow datasets
        self.setup_data_generators()

        # Save the splits if requested
        if save_splits and splits_dir:
            os.makedirs(splits_dir, exist_ok=True)

            splits_data = {
                'train_files': self.train_files,
                'train_labels': self.train_labels,
                'val_files': self.val_files,
                'val_labels': self.val_labels,
                'test_files': self.test_files,
                'test_labels': self.test_labels,
                'class_weights': self.class_weights
            }

            with open(os.path.join(splits_dir, 'data_splits.pkl'), 'wb') as f:
                pickle.dump(splits_data, f)

            # Save test files and labels as numpy arrays for evaluation
            np.save(os.path.join(splits_dir, 'test_files.npy'), np.array(self.test_files))
            np.save(os.path.join(splits_dir, 'test_labels.npy'), np.array(self.test_labels))

            print(f"Saved data splits to {splits_dir}")

        # Return dataset statistics
        return {
            'train_samples': len(self.train_files),
            'val_samples': len(self.val_files),
            'test_samples': len(self.test_files),
            'class_weights': self.class_weights
        }

    def load_from_splits(self, splits_dir):
        """
        Load previously saved data splits

        Args:
            splits_dir: Directory containing the saved splits

        Returns:
            True if successful, False otherwise
        """
        splits_path = os.path.join(splits_dir, 'data_splits.pkl')

        if not os.path.exists(splits_path):
            print(f"Data splits file not found at {splits_path}")
            return False

        try:
            with open(splits_path, 'rb') as f:
                splits_data = pickle.load(f)

            self.train_files = splits_data['train_files']
            self.train_labels = splits_data['train_labels']
            self.val_files = splits_data['val_files']
            self.val_labels = splits_data['val_labels']
            self.test_files = splits_data['test_files']
            self.test_labels = splits_data['test_labels']
            self.class_weights = splits_data['class_weights']

            # Create TensorFlow datasets
            self.setup_data_generators()

            print(f"Loaded data splits from {splits_dir}")
            print(f"Train set: {len(self.train_files)} samples")
            print(f"Validation set: {len(self.val_files)} samples")
            print(f"Test set: {len(self.test_files)} samples")

            return True
        except Exception as e:
            print(f"Error loading data splits: {e}")
            return False

    def setup_data_generators(self):
        """
        Setup TensorFlow datasets for training, validation, and testing
        """

        def generate_batches(files, labels, batch_size, is_training=False):
            """
            Create a TensorFlow dataset from file paths and labels.

            Args:
                files: List of file paths
                labels: List of integer labels (0 or 1)
                batch_size: Batch size for training
                is_training: Whether to apply data augmentation

            Returns:
                A TensorFlow dataset object
            """
            # Ensure all file paths are strings
            files = [str(f) for f in files]

            # Convert labels to a NumPy array with consistent dtype
            labels = np.array(labels, dtype=np.int32)

            # Create separate TensorFlow datasets for file paths and labels
            file_paths_ds = tf.data.Dataset.from_tensor_slices(files)
            labels_ds = tf.data.Dataset.from_tensor_slices(labels)

            # Zip the datasets together
            dataset = tf.data.Dataset.zip((file_paths_ds, labels_ds))

            # Function to load and preprocess images
            def load_and_preprocess_image(file_path, label):
                image = tf.io.read_file(file_path)
                image = tf.io.decode_png(image, channels=1)  # Ensure grayscale image
                image = tf.image.resize(image, [self.img_size, self.img_size])
                image = image / 255.0  # Normalize to [0,1]

                # # Apply augmentations if training
                # if is_training:
                #     image = tf.image.random_flip_left_right(image)
                #     image = tf.image.random_flip_up_down(image)
                #     # Add more augmentations if needed

                return image, label

            # Apply the processing function
            dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

            # Shuffle only during training
            if is_training:
                dataset = dataset.shuffle(buffer_size=len(files))

            # Batch and prefetch for efficiency
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            return dataset

        # Create TensorFlow datasets
        self.train_dataset = generate_batches(
            self.train_files, self.train_labels, self.batch_size, is_training=True)

        self.val_dataset = generate_batches(
            self.val_files, self.val_labels, 32)

        self.test_dataset = generate_batches(
            self.test_files, self.test_labels, 32)

    def load_test_data(self):
        """
        Load the test data as NumPy arrays for evaluation

        Returns:
            X_test: Test images as a NumPy array
            y_test: Test labels as a NumPy array
        """
        X_test = []

        for file_path in self.test_files:
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                X_test.append(img)
            except Exception as e:
                print(f"Error loading test image {file_path}: {e}")

        X_test = np.array(X_test) / 255.0  # Normalize
        y_test = np.array(self.test_labels)

        return X_test, y_test

    def load_and_balance_data(self, data_dir, malware_target=8500):
        """
        Loads benign and malware image files, and balances the malware samples by
        evenly sampling from each malware family.

        Args:
            data_dir (str): Data folder inside main directory.
            malware_target (int): Total number of malware samples desired.

        Returns:
            all_files (list): List of file paths for benign and balanced malware samples.
            labels (list): Binary labels (0 for benign, 1 for malware) corresponding to all_files.
        """
        benign_dir = os.path.join(data_dir, 'benign')
        malware_root = os.path.join(data_dir, 'malware')

        # Load benign samples
        benign_files = glob.glob(os.path.join(benign_dir, '*'))

        # Load malware samples from each family subfolder
        malware_families = [d for d in os.listdir(malware_root) if os.path.isdir(os.path.join(malware_root, d))]
        balanced_malware_files = []

        # For tracking family information
        family_info = []  # Will store family name for each sample

        # Determine number of samples to take per malware family
        samples_per_family = malware_target // len(malware_families)

        for family in malware_families:
            family_path = os.path.join(malware_root, family)
            family_files = glob.glob(os.path.join(family_path, '*'))
            # If not enough samples, take all; otherwise, sample uniformly
            if len(family_files) > samples_per_family:
                family_files = random.sample(family_files, samples_per_family)
            balanced_malware_files.extend(family_files)
            # Store family information for each file
            family_info.extend([family] * len(family_files))

        # Combine benign and malware files and create binary labels (0: benign, 1: malware)
        self.all_files = benign_files + balanced_malware_files
        self.all_labels = [0] * len(benign_files) + [1] * len(balanced_malware_files)

        # Store family information (benign samples get 'benign' as family)
        self.family_labels = ['benign'] * len(benign_files) + family_info

        # Create a mapping dictionary for quick lookup
        self.file_to_family_map = {file_path: family for file_path, family in zip(self.all_files, self.family_labels)}

        return self.all_files, self.all_labels

    def create_stratified_folds(self, files, binary_labels, n_splits=5, random_seed=None):
        """
        Creates stratified k-fold splits ensuring balanced distribution of
        both benign/malware classes and malware families.

        Args:
            files (list): List of file paths.
            binary_labels (list): Corresponding binary labels (0: benign, 1: malware).
            n_splits (int): Number of folds.
            random_seed (int): Random seed for fold creation (None for variable folds across runs)

        Returns:
            folds (list): A list of dictionaries with train/val/test indices.
        """
        # Use different random seed for each call if not specified
        if random_seed is None:
            random_seed = int(time.time()) % 10000  # Use current time as seed

        # Create stratification labels that include family information
        strat_labels = []
        for i, file_path in enumerate(files):
            if binary_labels[i] == 0:
                # Benign sample
                strat_labels.append(0)
            else:
                # Extract malware family from path
                # Assuming path structure: /data_dir/malware/family_name/sample.png
                parts = file_path.split(os.sep)
                family_idx = parts.index('malware') + 1
                if family_idx < len(parts):
                    family = parts[family_idx]
                    # Map each family to a unique integer starting from 1
                    family_id = self.family_to_id.get(family)
                    if family_id is None:
                        self.family_to_id[family] = len(self.family_to_id) + 1
                        family_id = self.family_to_id[family]
                    strat_labels.append(family_id)
                else:
                    # Fallback if path structure is unexpected
                    strat_labels.append(1)

        # Now use these stratification labels with StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        folds = []

        for train_val_idx, test_idx in skf.split(files, strat_labels):
            # Further split train_val into train and validation
            # Use stratified split here too
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.15 / (0.85),  # Adjusted to maintain overall 70/15/15 split
                stratify=[strat_labels[i] for i in train_val_idx],
                random_state=random_seed
            )

            folds.append({
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            })

        return folds
