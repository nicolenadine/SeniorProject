import os
from data_handler import DataHandler
from cross_validator import CrossValidator
from model_builder import ModelBuilder, train

def run_cross_validation_experiment(data_dir, output_dir, n_splits=5, epochs=50, img_size=256):
    """
    Run a complete cross-validation experiment.

    Args:
        data_dir: Directory containing malware and benign samples
        output_dir: Directory to save results
        n_splits: Number of folds for cross-validation
        epochs: Number of epochs per fold
        img_size: Image size for model input
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    data_handler = DataHandler(data_dir, img_size=img_size, batch_size=32)
    model_builder = ModelBuilder(img_size=img_size, model_type='resnet18', channels=1)

    # Initialize cross-validator
    cross_validator = CrossValidator(
        data_handler=data_handler,
        model_builder=model_builder,
        output_dir=output_dir,
        n_splits=n_splits,
        epochs=epochs,
        train_function=train
    )

    # Run cross-validation
    results = cross_validator.run_cross_validation()

    # Optionally, create ensemble of best models
    ensemble_dir = os.path.join(output_dir, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)

    # Save best models list for ensemble inference
    with open(os.path.join(ensemble_dir, 'ensemble_models.txt'), 'w') as f:
        for model_path in results['best_models']:
            f.write(f"{model_path}\n")

    print(f"\nCross-validation completed. Results saved to {output_dir}")

    # Display final results
    print("\nAverage Performance:")
    for metric, value in results['average_metrics'].items():
        std = results['std_metrics'][metric]
        print(f"{metric}: {value:.4f} ± {std:.4f}")