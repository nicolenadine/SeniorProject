#!/usr/bin/env python3
"""
Main script for the Malware Classification System
Provides a command-line interface to run the modular pipeline
"""

import os
import argparse
import tensorflow as tf
import json
from datetime import datetime
import numpy as np

# Import modules
from data_handler import DataHandler
from model_builder import ModelBuilder, CastLayer, train
from evaluator import Evaluator
from visualization import GradCAMGenerator
from cross_validator import CrossValidator
from visualization import HeatmapVisualizer
from feature_analyzer import FeatureAnalyzer
from error_analysis import ErrorAnalyzer
from comparative_metrics import ComparativeMetricsAnalyzer
from interactive_visualizations import InteractiveVisualizer
from visualization import analyze_by_family


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Malware Classification System')

    # Common arguments
    parser.add_argument('--data-dir', type=str, default='sample_images',
                        help='Directory containing malware and benign samples')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory to save results (default: timestamped directory)')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Size of the input images (img_size x img_size)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')

    # Cross-validation arguments
    parser.add_argument('--use-cross-validation', action='store_true',
                        help='Use cross-validation instead of single-fold training')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of folds for cross-validation')

    # Task selection argument
    parser.add_argument('task', type=str,
                        choices=['train', 'evaluate', 'gradcam', 'visualize',
                                 'cross-validation', 'feature-analysis',
                                 'error-analysis', 'metrics-analysis',
                                 'interactive-vis', 'family-gradcam', 'all'],
                        help='Task to perform')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs for training')
    parser.add_argument('--model-type', type=str, default='resnet18',
                        choices=['resnet18', 'simple_cnn'],
                        help='Type of model architecture to use')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducible fold creation')

    # Evaluation arguments
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a pre-trained model (required for evaluate/gradcam/visualize tasks)')
    parser.add_argument('--use-splits', action='store_true',
                        help='Use saved train/val/test splits')

    # Grad-CAM arguments
    parser.add_argument('--layer-name', type=str, default=None,
                        help='Name of the layer to use for Grad-CAM (default: last conv layer)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize per class')
    parser.add_argument('--generate-gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations during training/cross-validation')

    # Visualization arguments
    parser.add_argument('--heatmap-dir', type=str, default=None,
                        help='Directory containing heatmap files to visualize')

    # Feature analysis flag
    parser.add_argument('--run-feature-analysis', action='store_true',
                        help='Run feature importance analysis during cross-validation')

    return parser.parse_args()


def create_output_dir(args):
    """Create an output directory with timestamp if not specified"""
    if args.results_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.results_dir = f"results_{timestamp}"

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Using output directory: {args.results_dir}")

    return args.results_dir


def save_config(args, data_stats=None, output_dir=None):
    """Save configuration and dataset statistics"""
    if output_dir is None:
        output_dir = args.results_dir

    # Create a dictionary with all arguments
    config = vars(args).copy()

    # Add dataset statistics if available
    if data_stats:
        config.update(data_stats)

    # Save the configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {config_path}")


def evaluate(args, model=None, data_handler=None):
    """Evaluate a trained model"""
    print("=== Starting Evaluation Task ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Load model if not provided
    if model is None:
        custom_objects = {'CastLayer': CastLayer}

        # Check if model path is provided
        if args.model_path is None:
            # Try to find a model in the results directory
            if os.path.exists(os.path.join(args.results_dir, 'model')):
                model_dir = os.path.join(args.results_dir, 'model')
                # Check for final model first, then best model
                if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
                    args.model_path = os.path.join(model_dir, 'final_model.h5')
                elif os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                    args.model_path = os.path.join(model_dir, 'best_model.h5')

        if args.model_path is None:
            raise ValueError("Model path not provided and no model found in results directory")

        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    # Load data if not provided
    if data_handler is None:
        data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Try to load from splits first
        if args.use_splits and os.path.exists(os.path.join(args.results_dir, 'data_splits.pkl')):
            success = data_handler.load_from_splits(args.results_dir)
            if not success:
                data_handler.load_and_preprocess_data()
        else:
            data_handler.load_and_preprocess_data()

    # Load test data
    X_test, y_test = data_handler.load_test_data()

    # Initialize evaluator
    evaluator = Evaluator(model=model, output_dir=output_dir)

    # Evaluate model
    results = evaluator.evaluate(X_test, y_test)

    print(f"Evaluation completed. Results saved to {output_dir}")

    return results, X_test, y_test


def generate_gradcam(args, model=None, X_test=None, y_test=None):
    """Generate Grad-CAM visualizations"""
    print("=== Starting Grad-CAM Generation Task ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Load model if not provided
    if model is None:
        custom_objects = {'CastLayer': CastLayer}

        # Check if model path is provided
        if args.model_path is None:
            # Try to find a model in the results directory
            if os.path.exists(os.path.join(args.results_dir, 'model')):
                model_dir = os.path.join(args.results_dir, 'model')
                # Check for final model first, then best model
                if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
                    args.model_path = os.path.join(model_dir, 'final_model.h5')
                elif os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                    args.model_path = os.path.join(model_dir, 'best_model.h5')

        if args.model_path is None:
            raise ValueError("Model path not provided and no model found in results directory")

        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    # Load test data if not provided
    if X_test is None or y_test is None:
        data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Try to load from splits first
        if args.use_splits and os.path.exists(os.path.join(args.results_dir, 'data_splits.pkl')):
            success = data_handler.load_from_splits(args.results_dir)
            if not success:
                data_handler.load_and_preprocess_data()
        else:
            data_handler.load_and_preprocess_data()

        # Load test data
        X_test, y_test = data_handler.load_test_data()

    # Initialize Grad-CAM generator
    gradcam_generator = GradCAMGenerator(
        model=model,
        img_size=args.img_size,
        output_dir=output_dir
    )

    # Get layer names
    conv_layers = [layer.name for layer in model.layers
                   if isinstance(layer, tf.keras.layers.Conv2D)]

    # Generate heatmaps
    if args.layer_name is None:
        # If no layer name is provided, use the last 3 conv layers
        for layer_name in conv_layers[-3:]:
            heatmaps = gradcam_generator.generate_heatmaps(
                X_test, y_test, layer_name, num_classes=2
            )

            # Visualize heatmaps
            gradcam_generator.visualize_heatmaps(
                heatmaps, layer_name, class_names=['Benign', 'Malware']
            )

            # Create sample overlays
            gradcam_generator.create_sample_overlays(
                X_test, y_test, layer_name, num_samples=args.num_samples
            )
    else:
        # Use the specified layer
        heatmaps = gradcam_generator.generate_heatmaps(
            X_test, y_test, args.layer_name, num_classes=2
        )

        # Visualize heatmaps
        gradcam_generator.visualize_heatmaps(
            heatmaps, args.layer_name, class_names=['Benign', 'Malware']
        )

        # Create sample overlays
        gradcam_generator.create_sample_overlays(
            X_test, y_test, args.layer_name, num_samples=args.num_samples
        )

    print(f"Grad-CAM generation completed. Results saved to {output_dir}/gradcam/")


def visualize_heatmaps(args):
    """Visualize existing heatmap files"""
    print("=== Starting Heatmap Visualization Task ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Determine heatmap directory
    heatmap_dir = args.heatmap_dir
    if heatmap_dir is None:
        # Try to find the gradcam directory in the results directory
        if os.path.exists(os.path.join(args.results_dir, 'gradcam')):
            heatmap_dir = os.path.join(args.results_dir, 'gradcam')

    if heatmap_dir is None:
        raise ValueError("Heatmap directory not provided and no gradcam directory found in results directory")

    # Initialize heatmap visualizer
    visualizer = HeatmapVisualizer(output_dir=output_dir)

    # Batch visualize heatmaps
    visualizer.batch_visualize_heatmaps(heatmap_dir)

    # Create overlays for sample images
    if os.path.exists(args.data_dir):
        visualizer.batch_create_overlays(
            sample_images_dir=args.data_dir,
            heatmap_dir=heatmap_dir
        )

    print(f"Heatmap visualization completed. Results saved to {output_dir}/visualizations/")


def run_cross_validation(args):
    """Run k-fold cross-validation with comprehensive evaluation."""
    print("=== Starting Cross-Validation Task ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Initialize data handler
    data_handler = DataHandler(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    # Initialize model builder
    model_builder = ModelBuilder(
        img_size=args.img_size,
        model_type=args.model_type
    )

    # Add cross-validation specific arguments
    if not hasattr(args, 'n_splits'):
        args.n_splits = 5  # Default to 5 folds if not specified

    # Add visualization flag for gradcam
    if not hasattr(args, 'generate_gradcam'):
        args.generate_gradcam = True  # Default to generating gradcam visualizations

    # Initialize cross-validator
    cross_validator = CrossValidator(
        args=args,
        data_handler=data_handler,
        model_builder=model_builder,
        output_dir=output_dir,
        train_function=train(args,data_handler, model_builder, output_dir)  # Assuming `train` is a method of ModelBuilder
    )
    # Run cross-validation
    results = cross_validator.run_cross_validation()

    # Print summary results
    print("\nCross-Validation Summary:")
    for metric, value in results['aggregate_metrics'].items():
        if metric.endswith('_mean'):
            metric_name = metric[:-5]  # Remove '_mean' suffix
            std_value = results['aggregate_metrics'][f'{metric_name}_std']
            print(f"{metric_name}: {value:.4f} ± {std_value:.4f}")

    print(f"\nDetailed results saved to {output_dir}")

    return results


def analyze_features(args, model=None, data_handler=None):
    """Perform feature importance analysis on the trained model"""
    print("=== Starting Feature Analysis Task ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Load model if not provided
    if model is None:
        custom_objects = {'CastLayer': CastLayer}

        # Check if model path is provided
        if args.model_path is None:
            # Try to find a model in the results directory
            if os.path.exists(os.path.join(args.results_dir, 'model')):
                model_dir = os.path.join(args.results_dir, 'model')
                # Check for final model first, then best model
                if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
                    args.model_path = os.path.join(model_dir, 'final_model.h5')
                elif os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                    args.model_path = os.path.join(model_dir, 'best_model.h5')

        if args.model_path is None:
            raise ValueError("Model path not provided and no model found in results directory")

        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    # Load data if not provided
    if data_handler is None:
        data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Try to load from splits first
        if args.use_splits and os.path.exists(os.path.join(args.results_dir, 'data_splits.pkl')):
            success = data_handler.load_from_splits(args.results_dir)
            if not success:
                data_handler.load_and_preprocess_data()
        else:
            data_handler.load_and_preprocess_data()

    # Load test data
    X_test, y_test = data_handler.load_test_data()

    # Initialize feature analyzer
    analyzer = FeatureAnalyzer(
        model=model,
        data_handler=data_handler,
        output_dir=output_dir
    )

    # Create t-SNE visualization
    print("Creating t-SNE visualization...")
    analyzer.visualize_tsne(X_test, y_test)

    # Create UMAP visualization
    print("Creating UMAP visualization...")
    analyzer.visualize_umap(X_test, y_test)

    # Visualize layer activations for sample images
    print("Visualizing layer activations...")
    # Find the last few convolutional layers
    conv_layers = [layer.name for layer in model.layers
                   if isinstance(layer, tf.keras.layers.Conv2D)][-3:]

    # Use first few samples from each class
    class_samples = {}
    for i, label in enumerate(y_test):
        class_idx = int(label)
        if class_idx not in class_samples:
            class_samples[class_idx] = []
        if len(class_samples[class_idx]) < 3:  # Get 3 samples per class
            class_samples[class_idx].append(i)

    for class_idx, indices in class_samples.items():
        for idx in indices:
            analyzer.visualize_layer_activations(
                X_test[idx:idx + 1], layer_names=conv_layers, sample_index=0
            )

    # Generate class activation maps
    print("Generating class activation maps...")
    analyzer.generate_class_activation_maps(X_test, y_test, n_samples=5)

    # Visualize filter activations (activation maximization)
    print("Visualizing filter activations...")
    analyzer.visualize_activation_maximization()

    print(f"Feature analysis completed. Results saved to {output_dir}/feature_analysis/")

def analyze_errors(args, model=None, data_handler=None):
    """Perform error analysis on the trained model"""
    print("=== Starting Error Analysis Task ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Load model if not provided
    if model is None:
        custom_objects = {'CastLayer': CastLayer}

        # Check if model path is provided
        if args.model_path is None:
            # Try to find a model in the results directory
            if os.path.exists(os.path.join(args.results_dir, 'model')):
                model_dir = os.path.join(args.results_dir, 'model')
                # Check for final model first, then best model
                if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
                    args.model_path = os.path.join(model_dir, 'final_model.h5')
                elif os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                    args.model_path = os.path.join(model_dir, 'best_model.h5')

        if args.model_path is None:
            raise ValueError("Model path not provided and no model found in results directory")

        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    # Load data if not provided
    if data_handler is None:
        data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Try to load from splits first
        if args.use_splits and os.path.exists(os.path.join(args.results_dir, 'data_splits.pkl')):
            success = data_handler.load_from_splits(args.results_dir)
            if not success:
                data_handler.load_and_preprocess_data()
        else:
            data_handler.load_and_preprocess_data()

    # Load test data
    X_test, y_test = data_handler.load_test_data()

    # Initialize error analyzer
    analyzer = ErrorAnalyzer(
        model=model,
        data_handler=data_handler,
        output_dir=output_dir
    )

    # Perform comprehensive error analysis
    print("Performing error analysis...")
    class_names = ['Benign', 'Malware']  # Update with your class names
    results = analyzer.analyze_errors(X_test, y_test, class_names=class_names)

    print(f"Error analysis completed. Results saved to {output_dir}/error_analysis/")

    # Return a summary of the results
    error_rate = len(results['error_indices']) / len(y_test) * 100
    print(f"Error analysis summary:")
    print(f"- Total samples: {len(y_test)}")
    print(f"- Correct predictions: {len(results['correct_indices'])}")
    print(f"- Incorrect predictions: {len(results['error_indices'])}")
    print(f"- Error rate: {error_rate:.2f}%")

    # Print a basic classification report
    print("\nClassification Report:")
    for cls, metrics in results['classification_report'].items():
        if cls in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"Class: {cls}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']}")
    print(f"Overall Accuracy: {results['classification_report']['accuracy']:.4f}")

    return results

def analyze_metrics(args, model=None, data_handler=None, baseline_path=None):
    """Perform comprehensive comparative metrics analysis on the trained model"""
    print("=== Starting Comparative Metrics Analysis ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Load model if not provided
    if model is None:
        custom_objects = {'CastLayer': CastLayer}

        # Check if model path is provided
        if args.model_path is None:
            # Try to find a model in the results directory
            if os.path.exists(os.path.join(args.results_dir, 'model')):
                model_dir = os.path.join(args.results_dir, 'model')
                # Check for final model first, then best model
                if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
                    args.model_path = os.path.join(model_dir, 'final_model.h5')
                elif os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                    args.model_path = os.path.join(model_dir, 'best_model.h5')

        if args.model_path is None:
            raise ValueError("Model path not provided and no model found in results directory")

        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    # Load data if not provided
    if data_handler is None:
        data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Try to load from splits first
        if args.use_splits and os.path.exists(os.path.join(args.results_dir, 'data_splits.pkl')):
            success = data_handler.load_from_splits(args.results_dir)
            if not success:
                data_handler.load_and_preprocess_data()
        else:
            data_handler.load_and_preprocess_data()

    # Load test data
    X_test, y_test = data_handler.load_test_data()

    # Extract model name from path if available
    model_name = "current_model"
    if args.model_path:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Initialize metrics analyzer
    analyzer = ComparativeMetricsAnalyzer(
        model=model,
        data_handler=data_handler,
        output_dir=output_dir,
        model_name=model_name
    )

    # Compute comprehensive metrics
    print("Computing comprehensive metrics...")
    metrics_results = analyzer.compute_metrics(X_test, y_test)

    # Measure inference time performance
    print("Measuring inference time performance...")
    inference_results = analyzer.measure_inference_time(X_test[:100])  # Use a subset for speed

    # Compare with baseline if provided
    if baseline_path:
        print(f"Comparing with baseline model at {baseline_path}...")
        comparison_results = analyzer.compare_with_baseline(baseline_path, metrics_results)

        # Print key comparison insights
        print("\nModel Comparison Highlights:")
        for metric, values in comparison_results['metrics_comparison'].items():
            change = values['percent_change']
            direction = "improvement" if change > 0 else "decrease"
            print(f"- {metric}: {abs(change):.2f}% {direction}")

    # Try to generate per-family metrics if data is available
    try:
        # Check if family labels are available in data_handler
        if hasattr(data_handler, 'family_labels') and data_handler.family_labels is not None:
            print("Generating per-family performance metrics...")
            family_results = analyzer.generate_per_family_metrics(
                X_test, y_test, data_handler.family_labels
            )

            # Print per-family summary
            if family_results and 'per_family_metrics' in family_results:
                print("\nPer-Family Performance Summary:")
                families = list(family_results['per_family_metrics'].keys())
                f1_scores = [family_results['per_family_metrics'][f]['f1_score'] for f in families]
                best_family = families[np.argmax(f1_scores)]
                worst_family = families[np.argmin(f1_scores)]

                print(f"- Best performing family: {best_family} (F1: {max(f1_scores):.4f})")
                print(f"- Worst performing family: {worst_family} (F1: {min(f1_scores):.4f})")
                print(f"- Performance gap: {max(f1_scores) - min(f1_scores):.4f}")
    except Exception as e:
        print(f"Could not generate per-family metrics: {e}")
        print("Family-level analysis requires 'family_labels' in the data handler.")

    print(f"Comparative metrics analysis completed. Results saved to {output_dir}/comparative_metrics/")

    return metrics_results

def create_interactive_visualizations(args, model=None, data_handler=None):
    """Create interactive visualizations for analyzing model behavior"""
    print("=== Starting Interactive Visualization Creation ===")

    # Create output directory
    output_dir = create_output_dir(args)

    # Load model if not provided
    if model is None:
        custom_objects = {'CastLayer': CastLayer}

        # Check if model path is provided
        if args.model_path is None:
            # Try to find a model in the results directory
            if os.path.exists(os.path.join(args.results_dir, 'model')):
                model_dir = os.path.join(args.results_dir, 'model')
                # Check for final model first, then best model
                if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
                    args.model_path = os.path.join(model_dir, 'final_model.h5')
                elif os.path.exists(os.path.join(model_dir, 'best_model.h5')):
                    args.model_path = os.path.join(model_dir, 'best_model.h5')

        if args.model_path is None:
            raise ValueError("Model path not provided and no model found in results directory")

        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    # Load data if not provided
    if data_handler is None:
        data_handler = DataHandler(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        # Try to load from splits first
        if args.use_splits and os.path.exists(os.path.join(args.results_dir, 'data_splits.pkl')):
            success = data_handler.load_from_splits(args.results_dir)
            if not success:
                data_handler.load_and_preprocess_data()
        else:
            data_handler.load_and_preprocess_data()

    # Load test data
    X_test, y_test = data_handler.load_test_data()

    # Initialize interactive visualizer
    visualizer = InteractiveVisualizer(
        model=model,
        data_handler=data_handler,
        output_dir=output_dir
    )

    # Create various interactive visualizations
    print("Creating saliency slider dashboard...")
    saliency_path = visualizer.create_saliency_slider_dashboard(X_test, y_test)
    print(f"Saliency slider dashboard saved to {saliency_path}")

    print("Creating feature map explorer...")
    feature_map_path = visualizer.create_feature_map_explorer(X_test)
    print(f"Feature map explorer saved to {feature_map_path}")

    # Check if comparison model paths are provided
    if args.comparison_model_paths:
        print("Creating side-by-side comparison...")
        comparison_paths = args.comparison_model_paths.split(',')
        side_by_side_path = visualizer.create_side_by_side_comparison(comparison_paths, X_test, y_test)
        print(f"Side-by-side comparison saved to {side_by_side_path}")

    # Check if there are multiple results directories to compare
    if args.compare_results_dirs:
        print("Creating comparative dashboard...")
        results_dirs = args.compare_results_dirs.split(',')
        model_names = args.model_names.split(',') if args.model_names else None
        comparative_path = visualizer.create_comparative_dashboard(results_dirs, model_names)
        print(f"Comparative dashboard saved to {comparative_path}")

    print(f"Interactive visualizations created successfully. Results saved to {output_dir}/interactive_visualizations/")
    print(f"Open the HTML files in a web browser to use the interactive visualizations.")

    return {
        'saliency_dashboard': saliency_path,
        'feature_map_explorer': feature_map_path
    }


def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()

    # Set up TensorFlow GPU allocation settings (if needed)
    try:
        # Prevent TensorFlow from hogging all GPU memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
    except Exception as e:
        print(f"Error configuring GPUs: {e}")

    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Create output directory
    output_dir = create_output_dir(args)

    # Initialize data handler and model builder (needed for multiple tasks)
    data_handler = DataHandler(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    model_builder = ModelBuilder(
        img_size=args.img_size,
        model_type=args.model_type
    )

    # Execute the requested task
    if args.task == 'train':
        if hasattr(args, 'use_cross_validation') and args.use_cross_validation:
            # Run cross-validation training
            print("Running cross-validation training...")
            cv_results = run_cross_validation(args)

            # Print summary of the best fold
            best_fold_idx = np.argmax([e['metrics']['accuracy'] for e in cv_results['fold_evaluations']])
            best_fold_dir = os.path.join(args.results_dir, f"fold_{best_fold_idx + 1}")
            best_metrics = cv_results['fold_evaluations'][best_fold_idx]['metrics']

            print(f"\nBest fold: {best_fold_idx + 1}")
            print(f"Best fold directory: {best_fold_dir}")
            print("Best fold metrics:")
            for metric, value in best_metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            # Run single model training
            print("Running single model training...")
            model, data_handler = train(args, data_handler, model_builder, output_dir)
            print(f"Training completed. Model saved to {os.path.join(output_dir, 'model')}")

    elif args.task == 'evaluate':
        evaluate(args)

    elif args.task == 'gradcam':
        generate_gradcam(args)

    elif args.task == 'visualize':
        visualize_heatmaps(args)

    elif args.task == 'cross-validation':
        run_cross_validation(args)

    elif args.task == 'feature-analysis':
        analyze_features(args)

    elif args.task == 'error-analysis':
        analyze_errors(args)

    elif args.task == 'metrics-analysis':
        analyze_metrics(args, baseline_path=args.baseline_path if hasattr(args, 'baseline_path') else None)

    elif args.task == 'interactive-vis':
        create_interactive_visualizations(args)

    elif args.task == 'family-gradcam':
        # First, make sure to import the function if it's not already imported
        from visualization import analyze_by_family
        analyze_by_family(args)

    elif args.task == 'all':
        # Run all tasks in sequence with single model training (not cross-validation)
        print("Running all tasks with single model training...")

        # Run standard single-fold training and evaluation
        model, data_handler = train(args, data_handler, model_builder, output_dir)
        results, X_test, y_test = evaluate(args, model, data_handler)
        generate_gradcam(args, model, X_test, y_test)
        visualize_heatmaps(args)
        analyze_features(args, model, data_handler)
        analyze_errors(args, model, data_handler)
        analyze_metrics(args, model, data_handler, args.baseline_path if hasattr(args, 'baseline_path') else None)
        create_interactive_visualizations(args, model, data_handler)
        from visualization import analyze_by_family
        analyze_by_family(args, model, X_test, y_test, data_handler)


if __name__ == "__main__":
    main()