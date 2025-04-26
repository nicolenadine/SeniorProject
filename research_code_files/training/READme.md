# Malware Classification Pipeline: Core Training Files

## data_handler.py
**Purpose**: Handles data loading, preprocessing, and dataset creation for malware and benign image samples.

**Key Inputs**:
- `data_dir`: Directory containing 'malware' and 'benign' subdirectories
- `img_size`: Size to resize input images (default: 256x256)
- `batch_size`: Batch size for training

**Key Outputs**:
- `train_dataset`, `val_dataset`, `test_dataset`: TensorFlow datasets for training
- `family_labels`: Labels identifying malware families for each sample
- Data splits for cross-validation

## model_builder.py
**Purpose**: Defines and builds neural network architectures for malware classification.

**Key Inputs**:
- `img_size`: Input image dimensions
- `model_type`: Type of architecture to use (e.g., 'resnet18', 'simple_cnn')
- `channels`: Number of input channels (1 for grayscale)

**Key Outputs**:
- Compiled TensorFlow model
- List of convolutional layer names (for visualization)
- Training callbacks setup

## trainer.py
**Purpose**: Handles the training process for the malware classification models.

**Key Inputs**:
- `model`: TensorFlow model to train
- `data_handler`: Instance with prepared datasets
- `epochs`: Number of training epochs
- `callbacks`: List of training callbacks

**Key Outputs**:
- Training history (loss and metrics)
- Returns trained model

## main.py
**Purpose**: Central command-line interface script that coordinates all pipeline components.

**Key Inputs**:
- Command-line arguments for tasks (train, evaluate, visualize, etc.)
- Configuration for data, model, and training parameters

**Key Outputs**:
- Trained models
- Evaluation results
- Visualizations
- Coordinates execution of different pipeline components

## run_kfold_experiment.py
**Purpose**: Runs k-fold cross-validation experiments to validate model performance.

**Key Inputs**:
- `data_dir`: Data directory
- `output_dir`: Directory to save results
- `n_splits`: Number of folds for cross-validation
- `epochs`: Number of training epochs per fold

**Key Outputs**:
- Models trained on each fold
- Aggregated performance metrics
- Cross-validation summary

## cross_validator.py
**Purpose**: Implements comprehensive k-fold cross-validation with extensive analysis and visualization.

**Key Inputs**:
- `data_handler`: Prepared data handler
- `model_builder`: Model builder instance
- `n_splits`: Number of folds
- `train_function`: Function to train models

**Key Outputs**:
- Detailed per-fold evaluation results
- Aggregate metrics across folds
- Visualizations comparing performance across folds
- Family-based analysis results (if available)

## train_full_model.py
**Purpose**: Trains a model on full malware images with comprehensive data collection and analysis.

**Key Inputs**:
- `data_dir`: Directory containing malware and benign samples
- `results_dir`: Directory to save results
- Model and training parameters

**Key Outputs**:
- Trained model
- Detailed performance metrics
- GradCAM visualizations
- Training insights

## train_segmented_model.py
**Purpose**: Implements a segmented approach where images are divided into segments and an ensemble of models is trained.

**Key Inputs**:
- `data_dir`: Directory containing malware and benign samples
- `results_dir`: Directory to save results
- `segment_size`: Size of image segments
- `num_segments`: Number of segments per image
- Ensemble voting parameters

**Key Outputs**:
- Individual segment models
- Ensemble prediction performance
- Comparison with full-image model
- Segment-based GradCAM visualizations