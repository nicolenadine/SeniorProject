import os
import random  # Added for random fold processing
from model_builder import ModelBuilder, train
from evaluator import Evaluator
from visualization import GradCAMGenerator
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from training_insights import plot_metrics
from error_analysis import ErrorAnalyzer
from comparative_metrics import ComparativeMetricsAnalyzer


class CrossValidator:
    def __init__(self, args, data_handler, model_builder, output_dir, train_function):
        """Initialize cross-validation framework."""
        self.args = args
        self.data_handler = data_handler
        self.model_builder = model_builder
        self.output_dir = output_dir
        self.n_splits = args.n_splits if hasattr(args, 'n_splits') else 5
        self.epochs = args.epochs
        self.train_function = train_function
        self.random_seed = args.random_seed if hasattr(args, 'random_seed') else None

        # Create directories for aggregated results
        self.aggregate_dir = os.path.join(output_dir, 'aggregate_results')
        os.makedirs(self.aggregate_dir, exist_ok=True)

        # Storage for cross-validation results
        self.fold_metrics = []
        self.fold_histories = []
        self.fold_evaluations = []
        self.fold_family_evaluations = []

    def run_cross_validation(self):
        """Execute k-fold cross-validation with full evaluation."""
        # Load and preprocess data
        files, labels = self.data_handler.load_and_balance_data(
            self.data_handler.data_dir, malware_target=8500)

        # Store family labels (these were set in load_and_balance_data)
        family_labels = self.data_handler.family_labels

        # Create stratified folds
        folds = self.data_handler.create_stratified_folds(files, labels, self.n_splits, random_seed=self.random_seed)

        # Import for deep copying args
        import copy

        # Create a list of fold indices and optionally shuffle them to prevent sequential bias
        fold_indices = list(range(self.n_splits))
        if hasattr(self.args, 'shuffle_folds') and self.args.shuffle_folds:
            random.shuffle(fold_indices)

        # For each fold
        for fold_counter, fold_idx in enumerate(fold_indices):
            print(
                f"\n{'=' * 50}\nProcessing Fold {fold_idx + 1}/{self.n_splits} (order: {fold_counter + 1})\n{'=' * 50}")

            # Clear TensorFlow session to ensure independence between folds
            tf.keras.backend.clear_session()

            # Create fold directory
            fold_dir = os.path.join(self.output_dir, f"fold_{fold_idx + 1}")
            os.makedirs(fold_dir, exist_ok=True)

            # Create modified args for this fold
            fold_args = copy.deepcopy(self.args)
            fold_args.results_dir = fold_dir

            # Use the specified fold for training
            fold = folds[fold_idx]

            # Extract the test indices for reference later
            test_idx = fold['test_idx']

            # Extract family labels for test data if available
            test_families = [family_labels[idx] for idx in test_idx] if family_labels else None

            # Define train_val_families (all families not in test set)
            train_val_idx = np.concatenate([fold['train_idx'], fold['val_idx']])  # Combine train and val indices
            train_val_families = [family_labels[idx] for idx in train_val_idx] if family_labels else None

            # Pass these pre-created folds to your train function
            fold_args.precomputed_folds = {
                'folds': folds,
                'files': files,
                'labels': labels,
                'family_labels': family_labels
            }

            # Create a new model builder for each fold to prevent state leakage
            fold_model_builder = ModelBuilder(
                img_size=self.model_builder.img_size,
                model_type=self.args.model_type
            )

            # Train model on this fold - let it use the precomputed folds
            model, fold_data_handler = train(fold_args, self.data_handler, fold_model_builder, fold_dir,
                                             fold_index=fold_idx)

            # Load test data
            X_test, y_test = fold_data_handler.load_test_data()

            # Make sure family information is stored in data_handler for this fold
            if train_val_families and test_families:
                fold_data_handler.family_labels = train_val_families + test_families

            # Evaluate using your Evaluator class
            evaluator = Evaluator(model=model, output_dir=fold_dir)
            evaluation_results = evaluator.evaluate(X_test, y_test)
            self.fold_evaluations.append(evaluation_results)

            # NEW: Add family-specific analysis if family labels are available
            if hasattr(fold_data_handler, 'family_labels') and fold_data_handler.family_labels:
                # Get test set family labels
                test_family_labels = [fold_data_handler.family_labels[i]
                                      for i in range(len(fold_data_handler.test_files))]

                # Run error analysis by family
                error_analyzer = ErrorAnalyzer(model=model, data_handler=fold_data_handler, output_dir=fold_dir)
                family_error_results = error_analyzer.analyze_errors_by_family(X_test, y_test, test_family_labels)

                # Run metrics analysis by family
                metrics_analyzer = ComparativeMetricsAnalyzer(
                    model=model, data_handler=fold_data_handler, output_dir=fold_dir)
                family_metrics = metrics_analyzer.generate_per_family_metrics(X_test, y_test, test_family_labels)

                # Run comprehensive metrics analysis
                print(f"Generating comprehensive metrics for fold {fold_idx + 1}...")
                metrics_results = metrics_analyzer.compute_metrics(X_test, y_test)

                # Optionally measure inference time (can be slow)
                if hasattr(self.args, 'measure_inference') and self.args.measure_inference:
                    print(f"Measuring inference time for fold {fold_idx + 1}...")
                    inference_results = metrics_analyzer.measure_inference_time(X_test[:100])  # Use subset for speed

                # Store results for aggregation
                self.fold_family_evaluations.append({
                    'family_error_results': family_error_results,
                    'family_metrics': family_metrics,
                    'comprehensive_metrics': metrics_results
                })

            # Generate Grad-CAM visualizations if requested
            if hasattr(self.args, 'generate_gradcam') and self.args.generate_gradcam:
                self._generate_gradcam_for_fold(model, X_test, y_test, fold_dir, data_handler=fold_data_handler)

            # Save and process training history
            history_path = os.path.join(fold_dir, 'training_history.csv')
            if os.path.exists(history_path):
                history_df = pd.read_csv(history_path)
                self.fold_histories.append(history_df)

                # Generate training insights for this fold
                self._generate_insights_for_fold(history_df, fold_dir)

        # Generate aggregate results and visualizations
        self._generate_aggregate_results()

        return {
            'fold_evaluations': self.fold_evaluations,
            'fold_histories': self.fold_histories,
            'aggregate_metrics': self._calculate_aggregate_metrics(),
            'processing_order': fold_indices  # Added to track fold processing order
        }

    def _generate_gradcam_for_fold(self, model, X_test, y_test, fold_dir, data_handler=None):
        """
        Generate comprehensive Grad-CAM visualizations for a fold, including family-based analysis.

        Args:
            model: Trained model for this fold
            X_test: Test data for this fold
            y_test: Test labels for this fold
            fold_dir: Directory to save results for this fold
            data_handler: DataHandler instance with family labels (optional)
        """
        print(f"Generating Grad-CAM visualizations for fold...")

        # Initialize Grad-CAM generator
        gradcam_generator = GradCAMGenerator(
            model=model,
            img_size=self.args.img_size,
            output_dir=fold_dir
        )

        # Get convolutional layers
        conv_layers = [layer.name for layer in model.layers
                       if isinstance(layer, tf.keras.layers.Conv2D)]

        if not conv_layers:
            print("No convolutional layers found in the model. Skipping Grad-CAM generation.")
            return

        # Use last few convolutional layers or specified layer
        if hasattr(self.args, 'layer_name') and self.args.layer_name:
            target_layers = [self.args.layer_name]
        else:
            # Use last 1-3 convolutional layers
            target_layers = conv_layers[-min(3, len(conv_layers)):]

        results = {
            'class_heatmaps': {},
            'family_heatmaps': {}
        }

        for layer_name in target_layers:
            print(f"Analyzing layer: {layer_name}")

            # Step 1: Generate class-based heatmaps (benign vs. malware)
            print("Generating class-based heatmaps...")
            class_heatmaps = gradcam_generator.generate_heatmaps(
                X_test, y_test, layer_name, num_classes=2
            )

            # Visualize class-based heatmaps
            gradcam_generator.visualize_heatmaps(
                class_heatmaps, layer_name, class_names=['Benign', 'Malware']
            )

            # Create sample overlays
            print("Creating sample overlays...")
            gradcam_generator.create_sample_overlays(
                X_test, y_test, layer_name, num_samples=min(5, len(X_test))
            )

            # Create average overlays for each class
            print("Creating average overlays for benign and malware classes...")
            gradcam_generator.create_average_overlays(
                X_test, y_test, class_heatmaps, class_names=['Benign', 'Malware']
            )

            results['class_heatmaps'][layer_name] = class_heatmaps

            # Step 2: If family labels are available, generate family-specific heatmaps
            if data_handler and hasattr(data_handler, 'family_labels') and data_handler.family_labels:
                # Get test set family labels
                test_family_labels = []
                for i, file_path in enumerate(data_handler.test_files):
                    try:
                        orig_idx = data_handler.files.index(file_path)
                        test_family_labels.append(data_handler.family_labels[orig_idx])
                    except (ValueError, IndexError):
                        test_family_labels.append("unknown")

                print("Generating family-specific heatmaps...")
                family_heatmaps = gradcam_generator.generate_family_heatmaps(
                    X_test, y_test, test_family_labels, layer_name, num_classes=2
                )

                # Visualize family-specific heatmaps
                print("Visualizing family-specific heatmaps...")
                gradcam_generator.visualize_family_heatmaps(
                    family_heatmaps, layer_name
                )

                # Create sample overlays for each family
                print("Creating sample overlays for each malware family...")
                gradcam_generator.create_family_sample_overlays(
                    X_test, y_test, test_family_labels, layer_name,
                    num_samples=min(3, len(X_test))
                )

                # Create average overlays for each family
                print("Creating average overlays for each malware family...")
                gradcam_generator.create_family_average_overlays(
                    X_test, y_test, test_family_labels, family_heatmaps, layer_name
                )

                results['family_heatmaps'][layer_name] = family_heatmaps

        print(f"Grad-CAM generation completed for fold. Results saved to {fold_dir}/gradcam/")
        return results

    def _generate_insights_for_fold(self, history_df, fold_dir):
        """Generate training insights visualizations for this fold."""
        insights_dir = os.path.join(fold_dir, 'training_insights')
        os.makedirs(insights_dir, exist_ok=True)

        # Use your existing plot_metrics function
        plot_metrics(history_df, insights_dir)

    def _generate_aggregate_results(self):
        """Generate aggregate results across all folds."""
        # Create combined metrics DataFrame
        metrics_rows = []

        for fold_idx, evaluation in enumerate(self.fold_evaluations):
            row = {'Fold': fold_idx + 1}

            # Add metrics from evaluation
            for metric, value in evaluation['metrics'].items():
                row[metric] = value

            metrics_rows.append(row)

        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_rows)

        # Add summary row with means
        summary = {'Fold': 'Mean'}
        for col in metrics_df.columns:
            if col != 'Fold':
                summary[col] = metrics_df[col].mean()

        # Add standard deviation row
        std_row = {'Fold': 'Std Dev'}
        for col in metrics_df.columns:
            if col != 'Fold':
                std_row[col] = metrics_df[col].std()

        # Concatenate with summary rows
        metrics_df = pd.concat([metrics_df, pd.DataFrame([summary, std_row])])

        # Save to CSV
        metrics_df.to_csv(os.path.join(self.aggregate_dir, 'aggregate_metrics.csv'), index=False)

        # Generate comparison visualizations
        self._generate_comparison_visualizations()

    def _generate_comparison_visualizations(self):
        """Generate visualizations comparing performance across folds."""
        # Skip if we don't have enough fold data
        if len(self.fold_evaluations) < 2:
            return

        # 1. Accuracy and Loss Comparison
        fig = plt.figure(figsize=(12, 10))  # Create figure explicitly to fix the error

        # Create a DataFrame for easier plotting
        metrics_df = pd.DataFrame([
            {
                'Fold': fold_idx + 1,
                'Accuracy': eval_result['metrics']['accuracy'],
                'Loss': eval_result['metrics']['loss'],
                'AUC': eval_result['metrics'].get('auc', 0),
                'Precision': eval_result['metrics'].get('precision', 0),
                'Recall': eval_result['metrics'].get('recall', 0)
            }
            for fold_idx, eval_result in enumerate(self.fold_evaluations)
        ])

        # Plot accuracy comparison
        plt.subplot(2, 2, 1)
        sns.barplot(x='Fold', y='Accuracy', data=metrics_df)
        plt.axhline(y=metrics_df['Accuracy'].mean(), color='r', linestyle='--',
                    label=f'Mean: {metrics_df["Accuracy"].mean():.4f}')
        plt.title('Accuracy by Fold')
        plt.ylim(max(0, metrics_df['Accuracy'].min() - 0.1), 1.0)
        plt.legend()

        # Plot loss comparison
        plt.subplot(2, 2, 2)
        sns.barplot(x='Fold', y='Loss', data=metrics_df)
        plt.axhline(y=metrics_df['Loss'].mean(), color='r', linestyle='--',
                    label=f'Mean: {metrics_df["Loss"].mean():.4f}')
        plt.title('Loss by Fold')
        plt.legend()

        # Plot AUC comparison
        plt.subplot(2, 2, 3)
        if 'AUC' in metrics_df.columns:
            sns.barplot(x='Fold', y='AUC', data=metrics_df)
            plt.axhline(y=metrics_df['AUC'].mean(), color='r', linestyle='--',
                        label=f'Mean: {metrics_df["AUC"].mean():.4f}')
            plt.title('AUC by Fold')
            plt.ylim(max(0, metrics_df['AUC'].min() - 0.1), 1.0)
            plt.legend()

        # Plot Precision vs Recall
        plt.subplot(2, 2, 4)
        if 'Precision' in metrics_df.columns and 'Recall' in metrics_df.columns:
            width = 0.35
            x = np.arange(len(metrics_df))
            plt.bar(x - width / 2, metrics_df['Precision'], width, label='Precision')
            plt.bar(x + width / 2, metrics_df['Recall'], width, label='Recall')
            plt.xlabel('Fold')
            plt.title('Precision vs Recall by Fold')
            plt.xticks(x, metrics_df['Fold'])
            plt.ylim(max(0, min(metrics_df['Precision'].min(), metrics_df['Recall'].min()) - 0.1), 1.0)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.aggregate_dir, 'cross_validation_comparison.png'), dpi=300)
        plt.close()

        # 2. Training Curves Comparison (if histories are available)
        if self.fold_histories:
            self._generate_training_curve_comparison()

    def _generate_training_curve_comparison(self):
        """Generate comparison of training curves across folds."""
        fig = plt.figure(figsize=(15, 10))  # Create figure explicitly

        # Accuracy comparison
        plt.subplot(2, 2, 1)
        for fold_idx, history_df in enumerate(self.fold_histories):
            plt.plot(history_df['val_accuracy'], label=f'Fold {fold_idx + 1}')
        plt.title('Validation Accuracy Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Loss comparison
        plt.subplot(2, 2, 2)
        for fold_idx, history_df in enumerate(self.fold_histories):
            plt.plot(history_df['val_loss'], label=f'Fold {fold_idx + 1}')
        plt.title('Validation Loss Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # AUC comparison (if available)
        plt.subplot(2, 2, 3)
        auc_available = all('val_auc' in history_df.columns for history_df in self.fold_histories)
        if auc_available:
            for fold_idx, history_df in enumerate(self.fold_histories):
                plt.plot(history_df['val_auc'], label=f'Fold {fold_idx + 1}')
            plt.title('Validation AUC Across Folds')
            plt.xlabel('Epoch')
            plt.ylabel('Validation AUC')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

        # Training time comparison (if epochs vary)
        plt.subplot(2, 2, 4)
        epoch_counts = [len(history_df) for history_df in self.fold_histories]
        plt.bar(range(1, len(epoch_counts) + 1), epoch_counts)
        plt.axhline(y=np.mean(epoch_counts), color='r', linestyle='--',
                    label=f'Mean: {np.mean(epoch_counts):.1f}')
        plt.title('Training Duration by Fold')
        plt.xlabel('Fold')
        plt.ylabel('Epochs')
        plt.xticks(range(1, len(epoch_counts) + 1))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.aggregate_dir, 'training_curves_comparison.png'), dpi=300)
        plt.close()

    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all folds."""
        # Extract metrics from evaluations
        metrics_dict = {}

        # Get all metric names
        if self.fold_evaluations:
            metric_names = list(self.fold_evaluations[0]['metrics'].keys())

            # Calculate mean and std for each metric
            for metric in metric_names:
                values = [eval_result['metrics'][metric] for eval_result in self.fold_evaluations]
                metrics_dict[f'{metric}_mean'] = np.mean(values)
                metrics_dict[f'{metric}_std'] = np.std(values)

        return metrics_dict

    def _save_aggregate_results(self, results):
        """Save aggregate cross-validation results"""
        # Create summary dataframe
        summary_rows = []

        # Add individual fold results
        for fold_idx, metrics in enumerate(results['fold_metrics']):
            row = {'Fold': f'Fold {fold_idx + 1}'}
            row.update(metrics)
            summary_rows.append(row)

        # Add average row
        avg_row = {'Fold': 'Average'}
        avg_row.update(results['aggregate_metrics'])
        summary_rows.append(avg_row)

        # Create dataframe and save to CSV
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(self.output_dir, 'cross_validation_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"Cross-validation summary saved to {csv_path}")

        # Save complete results as JSON
        import json

        # Convert numpy values to Python native types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Create a serializable version of the results
        serializable_results = {
            'fold_metrics': [[convert_to_serializable(v) for k, v in metrics.items()]
                             for metrics in results['fold_metrics']],
            'average_metrics': {k: convert_to_serializable(v) for k, v in results['average_metrics'].items()},
            'std_metrics': {k: convert_to_serializable(v) for k, v in results['std_metrics'].items()},
            'best_models': results['best_models'],
            'processing_order': results.get('processing_order', list(range(len(results['fold_metrics']))))
        }

        # Save JSON
        json_path = os.path.join(self.output_dir, 'cross_validation_results.json')
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)