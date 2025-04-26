import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from model_builder import CastLayer, focal_loss
from visualization import *
from feature_analyzer import FeatureAnalyzer
import os
import json

# Set the file paths
metadata_file = 'results/segment_model/segment_selection_metadata.json'
test_predictions_file = 'results/segment_model/test_predictions.npy'
test_probabilities_file = 'results/segment_model/test_probabilities.npy'

# Check if files exist
if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

# Load the metadata
print(f"Loading metadata from: {metadata_file}")
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Create dataframes from each part of the metadata
train_df = pd.DataFrame(metadata['train'])
val_df = pd.DataFrame(metadata['validation'])
test_df = pd.DataFrame(metadata['test'])

# Print info about each dataframe
print(f"Train dataframe shape: {train_df.shape}")
print(f"Validation dataframe shape: {val_df.shape}")
print(f"Test dataframe shape: {test_df.shape}")

# Combine all data for the overall analysis
df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Combined dataframe shape: {df.shape}")

# ------- PART 1: OVERALL SEGMENT SELECTION ANALYSIS -------

# Visualize segment selection by class (Benign/Malware)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='selected_segment', hue='label')
plt.title("Segment Selection Frequency (Benign vs Malware)")
plt.xlabel("Segment Index (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)")
plt.ylabel("Count")
plt.legend(title='Class', labels=['Benign', 'Malware'])
plt.xticks([0, 1, 2, 3])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('segment_freq_by_class.png', dpi=300)
plt.show()

# Visualize segment selection by malware family
malware_df = df[df['label'] == 1]  # Malware only

# Check if we have enough malware samples
print(f"Number of malware samples: {len(malware_df)}")

if len(malware_df) > 0:
    # Get top 10 most common families to keep the plot readable
    top_families = malware_df['family'].value_counts().nlargest(10).index.tolist()
    print(f"Top 10 malware families: {top_families}")

    plot_df = malware_df[malware_df['family'].isin(top_families)]

    plt.figure(figsize=(14, 8))
    sns.countplot(data=plot_df, x='selected_segment', hue='family')
    plt.title("Segment Selection by Top 10 Malware Families")
    plt.xlabel("Segment Index (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)")
    plt.ylabel("Count")
    plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([0, 1, 2, 3])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('segment_freq_by_family.png', dpi=300)
    plt.show()
else:
    print("No malware samples found in the dataset. Skipping malware family visualization.")

# Create a heatmap of segment variance by class
# First, verify the structure of segment_variances
sample_variance = df['segment_variances'].iloc[0]
print(f"Sample segment_variances: {sample_variance}, type: {type(sample_variance)}")

# Create the variance dataframe
variance_dfs = []
for idx, row in df.iterrows():
    try:
        variances = row['segment_variances']
        if isinstance(variances, list) and len(variances) == 4:
            variance_df = pd.DataFrame({
                'label': row['label'],
                'family': row['family'],
                'segment_0_variance': variances[0],
                'segment_1_variance': variances[1],
                'segment_2_variance': variances[2],
                'segment_3_variance': variances[3],
                'selected_segment': row['selected_segment']
            }, index=[idx])
            variance_dfs.append(variance_df)
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        print(f"Row data: {row}")

# Only proceed if we have variance data
if variance_dfs:
    # Combine all variance data
    variance_data = pd.concat(variance_dfs)

    # Calculate average variance for each segment by class
    avg_variance_by_class = variance_data.groupby('label')[[
        'segment_0_variance', 'segment_1_variance', 'segment_2_variance', 'segment_3_variance'
    ]].mean()

    # Check if we have both classes
    if len(avg_variance_by_class) == 2:
        # Rename for better visualization
        avg_variance_by_class.index = ['Benign', 'Malware']
    else:
        # Use the labels we have
        avg_variance_by_class.index = [f"Class {label}" for label in avg_variance_by_class.index]

    avg_variance_by_class.columns = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(avg_variance_by_class, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Average Variance by Image Segment and Class')
    plt.tight_layout()
    plt.savefig('segment_variance_heatmap.png', dpi=300)
    plt.show()

    # Additional: distribution of variances for each segment by class
    plt.figure(figsize=(14, 10))
    for i, segment in enumerate(
            ['segment_0_variance', 'segment_1_variance', 'segment_2_variance', 'segment_3_variance']):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(data=variance_data, x='label', y=segment)
        plt.title(f'Variance Distribution - {["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"][i]}')
        plt.xlabel('Class')
        plt.ylabel('Variance')
        plt.xticks([0, 1], ['Benign', 'Malware'])
    plt.tight_layout()
    plt.savefig('segment_variance_distributions.png', dpi=300)
    plt.show()
else:
    print("Could not process segment variance data. Skipping heatmap.")

# ------- PART 2: MODEL PERFORMANCE ANALYSIS BY SEGMENT -------

# Check if prediction files exist for performance analysis
if os.path.exists(test_predictions_file) and os.path.exists(test_probabilities_file):
    # Load model predictions
    test_predictions = np.load(test_predictions_file)
    test_probabilities = np.load(test_probabilities_file)

    # Get test data from metadata
    test_data = pd.DataFrame(metadata['test'])

    # Ensure predictions match test data length
    if len(test_predictions) != len(test_data):
        print(
            f"Warning: Test predictions length ({len(test_predictions)}) doesn't match test data length ({len(test_data)})")
        # Use the minimum length to avoid errors
        min_length = min(len(test_predictions), len(test_data))
        test_predictions = test_predictions[:min_length]
        test_data = test_data.iloc[:min_length]

    # Add predictions to test data
    test_data['predicted'] = test_predictions
    test_data['probability'] = test_probabilities

    # Create a confusion matrix for each segment
    segment_names = ['Top-Left (0)', 'Top-Right (1)', 'Bottom-Left (2)', 'Bottom-Right (3)']
    segment_cms = []

    plt.figure(figsize=(16, 12))

    for segment_idx in range(4):
        # Filter data for this segment
        segment_data = test_data[test_data['selected_segment'] == segment_idx]

        if len(segment_data) > 0:
            # Get actual and predicted labels
            y_true = segment_data['label']
            y_pred = segment_data['predicted']

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Calculate accuracy for this segment
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)

            # Create subplot
            plt.subplot(2, 2, segment_idx + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Benign', 'Malware'],
                        yticklabels=['Benign', 'Malware'])
            plt.title(f'{segment_names[segment_idx]}\nAccuracy: {accuracy:.4f} (n={len(segment_data)})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save segment performance for comparison
            segment_cms.append({
                'segment': segment_idx,
                'name': segment_names[segment_idx],
                'accuracy': accuracy,
                'sample_count': len(segment_data),
                'confusion_matrix': cm
            })
        else:
            plt.subplot(2, 2, segment_idx + 1)
            plt.text(0.5, 0.5, f"No samples with segment {segment_idx}",
                     horizontalalignment='center', verticalalignment='center')
            plt.title(segment_names[segment_idx])

    plt.tight_layout()
    plt.savefig('segment_confusion_matrices.png', dpi=300)
    plt.show()

    # Create a summary bar chart comparing accuracy by segment
    if segment_cms:
        segment_df = pd.DataFrame(segment_cms)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(segment_df['name'], segment_df['accuracy'], alpha=0.7)

        # Add sample counts above each bar
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"n={segment_df['sample_count'][i]}",
                     ha='center', va='bottom', fontsize=9)

        plt.title('Model Accuracy by Selected Segment')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)  # Set y-axis from 0 to 1.1 to make room for sample counts
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('segment_accuracy_comparison.png', dpi=300)
        plt.show()

    # Additional analysis: Error analysis by segment
    # Create a visualization showing which segments have the most errors
    test_data['correct'] = test_data['label'] == test_data['predicted']

    plt.figure(figsize=(12, 5))

    # Subplot 1: Correct vs incorrect predictions by segment
    plt.subplot(1, 2, 1)
    correct_counts = pd.crosstab(test_data['selected_segment'], test_data['correct'])
    correct_counts.columns = ['Incorrect', 'Correct']
    correct_counts.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#ff9999', '#66b3ff'])
    plt.title('Prediction Results by Segment')
    plt.xlabel('Segment Index')
    plt.ylabel('Count')
    plt.xticks(range(4), segment_names, rotation=45)
    plt.legend(title='Prediction')

    # Subplot 2: Error rate by segment
    plt.subplot(1, 2, 2)
    error_rates = []
    for seg in range(4):
        seg_data = test_data[test_data['selected_segment'] == seg]
        if len(seg_data) > 0:
            error_rate = 1 - seg_data['correct'].mean()
            error_rates.append(error_rate)
        else:
            error_rates.append(0)

    plt.bar(range(4), error_rates, color='#ff9999')
    plt.title('Error Rate by Segment')
    plt.xlabel('Segment Index')
    plt.ylabel('Error Rate')
    plt.xticks(range(4), segment_names, rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('segment_error_analysis.png', dpi=300)
    plt.show()

    # Print a summary report
    print("Segment Performance Summary:")
    print("=" * 50)
    for segment in segment_cms:
        print(f"Segment: {segment['name']}")
        print(f"  Accuracy: {segment['accuracy']:.4f}")
        print(f"  Sample Count: {segment['sample_count']}")
        cm = segment['confusion_matrix']
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
        print("-" * 50)
else:
    print("Prediction files not found. Skipping model performance analysis.")
    print("To analyze model performance, make sure these files exist:")
    print(f"  - {test_predictions_file}")
    print(f"  - {test_probabilities_file}")

# --------- Variance distribution by family (alternative visualization)
# First, let's create a heatmap of average variances

# Convert the segment variances to a more usable format
family_segment_variances = {}

for _, row in df.iterrows():
    family = row['family']
    if family not in family_segment_variances:
        family_segment_variances[family] = [[], [], [], []]

    for i, variance in enumerate(row['segment_variances']):
        family_segment_variances[family][i].append(variance)

# Calculate average variance for each family and segment
avg_variances = {}
for family, segments in family_segment_variances.items():
    avg_variances[family] = [np.mean(segment) for segment in segments]

# Convert to DataFrame for visualization
heatmap_data = pd.DataFrame(avg_variances).T
heatmap_data.columns = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']

# Filter to top 10 families to keep visualization manageable
top_families = df['family'].value_counts().nlargest(10).index.tolist()
heatmap_data = heatmap_data.loc[top_families]

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='viridis')
plt.title('Average Variance by Family and Image Segment')
plt.tight_layout()
plt.savefig('family_variance_heatmap.png', dpi=300)
plt.show()

# Now, let's visualize segment selection frequency by family
# Count how often each segment is selected for each family
segment_counts = df.groupby(['family', 'selected_segment']).size().unstack(fill_value=0)
segment_counts.columns = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']

# Convert to percentages
segment_percent = segment_counts.div(segment_counts.sum(axis=1), axis=0) * 100

# Filter to top 10 families
segment_percent = segment_percent.loc[top_families]

# Plot as stacked bar chart
plt.figure(figsize=(14, 8))
segment_percent.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Segment Selection Frequency by Family (%)')
plt.xlabel('Family')
plt.ylabel('Percentage')
plt.legend(title='Selected Segment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('family_segment_selection.png', dpi=300)
plt.show()


#----------------------------------------------------
def analyze_segments_by_family(args, model=None, data_handler=None):
    """Generate Grad-CAM visualizations for segment model grouped by malware family"""
    print("=== Starting Segment-Based Family Grad-CAM Analysis ===")

    # Load metadata from the segment selection
    metadata_file = os.path.join(args.results_dir, 'segment_selection_metadata.json')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load model if not provided
    if model is None:
        custom_objects = {
            'CastLayer': CastLayer,
            'focal_loss_fixed': focal_loss
        }
        model_path = os.path.join(args.results_dir, 'model', 'best_model.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.results_dir, 'final_segment_model.keras')

        if not os.path.exists(model_path):
            raise ValueError(f"No model found at {model_path}")

        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Get test data from metadata
    test_metadata = metadata['test']
    test_files = [record['file_path'] for record in test_metadata]
    test_labels = [record['label'] for record in test_metadata]
    test_families = [record['family'] for record in test_metadata]
    selected_segments = [record['selected_segment'] for record in test_metadata]

    # Load and prepare segment test data
    X_test = []
    for i, file_path in enumerate(test_files):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))

        # Extract the selected segment
        seg_idx = selected_segments[i]
        if seg_idx == 0:  # top-left
            segment = img[:128, :128]
        elif seg_idx == 1:  # top-right
            segment = img[:128, 128:]
        elif seg_idx == 2:  # bottom-left
            segment = img[128:, :128]
        else:  # bottom-right
            segment = img[128:, 128:]

        # Normalize and add channel dimension
        segment = segment[..., np.newaxis] / 255.0
        X_test.append(segment)

    X_test = np.array(X_test)
    y_test = np.array(test_labels)

    # Initialize Grad-CAM generator
    gradcam_generator = GradCAMGenerator(
        model=model,
        img_size=128,  # Note: segment size is 128x128
        output_dir=args.results_dir
    )

    # Get layer names
    conv_layers = [layer.name for layer in model.layers
                   if isinstance(layer, tf.keras.layers.Conv2D)]

    # If no specific layer is provided, use the last few conv layers
    target_layers = [args.layer_name] if hasattr(args, 'layer_name') and args.layer_name else conv_layers[-3:]

    class_heatmaps = {}
    family_heatmaps = {}

    for layer_name in target_layers:
        print(f"Analyzing layer: {layer_name}")

        # Generate standard heatmaps by class
        print("Generating class-based heatmaps...")
        current_class_heatmaps = gradcam_generator.generate_heatmaps(
            X_test, y_test, layer_name, num_classes=2
        )

        # Store current layer's class heatmaps in results
        class_heatmaps[layer_name] = current_class_heatmaps

        # Visualize standard heatmaps
        gradcam_generator.visualize_heatmaps(
            current_class_heatmaps, layer_name, class_names=['Benign', 'Malware']
        )

        # Generate family-based heatmaps
        print("Generating family-based heatmaps...")
        current_family_heatmaps = gradcam_generator.generate_family_heatmaps(
            X_test, y_test, test_families, layer_name, num_classes=2
        )

        # Store current layer's family heatmaps in results
        family_heatmaps[layer_name] = current_family_heatmaps

        # Visualize family heatmaps
        print("Visualizing family-based heatmaps...")
        gradcam_generator.visualize_family_heatmaps(
            current_family_heatmaps, layer_name
        )

        # Create sample overlays for each family
        print("Creating sample overlays for each family...")
        gradcam_generator.create_family_sample_overlays(
            X_test, y_test, test_families, layer_name,
            num_samples=min(3, len(X_test))
        )

        # Create average overlays for each family
        print("Creating average overlays for each family...")
        gradcam_generator.create_family_average_overlays(
            X_test, y_test, test_families, current_family_heatmaps, layer_name
        )

    print(f"Segment-based family Grad-CAM analysis completed. Results saved to {args.results_dir}/gradcam/")
    return {
        'class_heatmaps': class_heatmaps,
        'family_heatmaps': family_heatmaps
    }


def analyze_segment_features(args, model=None):
    """Perform feature importance analysis on the segment-based model"""
    print("=== Starting Segment Feature Analysis ===")

    # Load metadata
    metadata_file = os.path.join(args.results_dir, 'segment_selection_metadata.json')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load model if not provided
    if model is None:
        custom_objects = {
            'CastLayer': CastLayer,
            'focal_loss_fixed': focal_loss
        }
        model_path = os.path.join(args.results_dir, 'model', 'best_model.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.results_dir, 'final_segment_model.keras')

        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    # Prepare test data from metadata
    test_metadata = metadata['test']
    X_test = []
    y_test = []
    family_labels = []
    segment_indices = []

    # Load segment data
    for record in test_metadata:
        img = cv2.imread(record['file_path'], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))

        # Extract the selected segment
        seg_idx = record['selected_segment']
        if seg_idx == 0:  # top-left
            segment = img[:128, :128]
        elif seg_idx == 1:  # top-right
            segment = img[:128, 128:]
        elif seg_idx == 2:  # bottom-left
            segment = img[128:, :128]
        else:  # bottom-right
            segment = img[128:, 128:]

        # Normalize and add channel dimension
        segment = segment[..., np.newaxis] / 255.0
        X_test.append(segment)
        y_test.append(record['label'])
        family_labels.append(record['family'])
        segment_indices.append(seg_idx)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Initialize feature analyzer
    analyzer = FeatureAnalyzer(
        model=model,
        output_dir=args.results_dir
    )

    # Create t-SNE visualization with segment information
    print("Creating t-SNE visualization with segment information...")
    analyzer.visualize_tsne(X_test, y_test, metadata=segment_indices,
                            metadata_name="Segment Index",
                            label_names=['Benign', 'Malware'])

    # Create UMAP visualization with segment information
    print("Creating UMAP visualization with segment information...")
    analyzer.visualize_umap(X_test, y_test, metadata=segment_indices,
                            metadata_name="Segment Index",
                            label_names=['Benign', 'Malware'])

    # Visualize layer activations for sample images from each segment
    print("Visualizing layer activations by segment...")

    # Find the last few convolutional layers
    conv_layers = [layer.name for layer in model.layers
                   if isinstance(layer, tf.keras.layers.Conv2D)][-3:]

    # Create a dictionary to store samples by segment and class
    segment_class_samples = {}
    for i, (label, seg_idx) in enumerate(zip(y_test, segment_indices)):
        key = (int(seg_idx), int(label))
        if key not in segment_class_samples:
            segment_class_samples[key] = []
        if len(segment_class_samples[key]) < 2:  # Get 2 samples per segment-class combination
            segment_class_samples[key].append(i)

    # Visualize activations for each sample
    for (seg_idx, class_idx), indices in segment_class_samples.items():
        segment_name = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right'][seg_idx]
        class_name = ['Benign', 'Malware'][class_idx]

        for idx in indices:
            analyzer.visualize_layer_activations(
                X_test[idx:idx + 1],
                layer_names=conv_layers,
                sample_index=0,
                title_prefix=f"{segment_name} Segment - {class_name}"
            )

    print(f"Segment feature analysis completed. Results saved to {args.results_dir}/feature_analysis/")

#------------------------------------------------------------------
# Add your analyze_segments_by_family and analyze_segment_features functions here

# Simple main block for running directly in IDE
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Segment Visualization')
    parser.add_argument('--analysis', type=str, choices=['gradcam', 'features', 'basic', 'all'],
                        default='basic', help='Type of analysis to perform')
    parser.add_argument('--results-dir', type=str, default='results/segment_model',
                        help='Directory containing segment model results')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='Name of layer to use for GradCAM (default: last conv layer)')

    args = parser.parse_args()

    # Set up directories and file paths
    metadata_file = os.path.join(args.results_dir, 'segment_selection_metadata.json')
    test_predictions_file = os.path.join(args.results_dir, 'test_predictions.npy')
    test_probabilities_file = os.path.join(args.results_dir, 'test_probabilities.npy')

    # Basic visualizations (what your current script does)
    if args.analysis == 'basic' or args.analysis == 'all':
        print("Running basic segment analysis...")
        # Your existing code for basic visualizations

    # GradCAM analysis
    if args.analysis == 'gradcam' or args.analysis == 'all':
        print("Running segment GradCAM analysis...")
        analyze_segments_by_family(args)

    # Feature analysis
    if args.analysis == 'features' or args.analysis == 'all':
        print("Running segment feature analysis...")
        analyze_segment_features(args)

