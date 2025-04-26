# Malware Classification Pipeline: Testing, Analysis & Visualization Files

## evaluator.py
**Purpose**: Evaluates model performance with comprehensive metrics and visualizations.

**Key Inputs**:
- `model`: Trained model to evaluate
- `X_test`, `y_test`: Test data and labels
- `output_dir`: Directory to save evaluation results

**Key Outputs**:
- Classification metrics (accuracy, precision, recall, F1, etc.)
- Confusion matrix visualization
- ROC curves
- Performance metrics in various formats (JSON, CSV)

## visualization.py
**Purpose**: Provides interpretability through Grad-CAM visualizations showing which image regions influence model decisions.

**Key Inputs**:
- `model`: Trained model to visualize
- `X_data`, `y_data`: Input data and labels
- `layer_name`: Name of layer to use for Grad-CAM
- `output_dir`: Directory to save visualizations

**Key Outputs**:
- Grad-CAM heatmaps highlighting influential image regions
- Class activation maps
- Per-family visualizations
- Sample overlay visualizations

## feature_analyzer.py
**Purpose**: Analyzes feature importance and visualizes high-dimensional embeddings from the model.

**Key Inputs**:
- `model`: Trained model to analyze
- `X_data`, `y_data`: Input data and labels
- `output_dir`: Directory to save analysis results

**Key Outputs**:
- t-SNE visualizations of embeddings
- UMAP visualizations
- Layer activation visualizations
- Activation maximization (filter visualization)

## error_analysis.py
**Purpose**: Performs detailed analysis of model errors to understand failure points.

**Key Inputs**:
- `model`: Trained model to analyze
- `X_data`, `y_true`: Input data and true labels
- `output_dir`: Directory to save analysis results

**Key Outputs**:
- Analysis of most confident errors
- Error distribution visualization
- Boundary case examples
- Per-family error analysis

## comparative_metrics.py
**Purpose**: Compares multiple models across various performance metrics.

**Key Inputs**:
- `model`: Primary model to analyze
- `data_handler`: DataHandler with test data
- `baseline_path`: Path to baseline model for comparison
- `output_dir`: Directory to save results

**Key Outputs**:
- Detailed performance comparison metrics
- ROC and precision-recall curves
- Optimal threshold analysis
- Per-family performance comparisons

## training_insights.py
**Purpose**: Generates training process visualizations to understand model learning behavior.

**Key Inputs**:
- `history_df`: Training history DataFrame with metrics
- `output_dir`: Directory to save visualizations

**Key Outputs**:
- Learning curves (accuracy, loss)
- Overfitting analysis
- Metric correlation visualizations
- Training recommendations

## training_insights_extended.py
**Purpose**: Provides advanced analysis of training dynamics with statistical measures.

**Key Inputs**:
- Same as training_insights.py plus additional statistical capabilities

**Key Outputs**:
- Moving averages of metrics
- Correlation heatmaps
- Metric variability analysis
- Comprehensive training report

## segment_analysis.py
**Purpose**: Analyzes the segmented model approach to understand which image regions are most informative.

**Key Inputs**:
- `results_dir`: Directory with segment model results
- `metadata_file`: Segment selection metadata

**Key Outputs**:
- Segment selection frequency visualization
- Variance analysis by segment
- Family-based segment importance
- Per-segment performance analysis

## ks_test.py
**Purpose**: Performs Kolmogorov-Smirnov tests to compare probability distributions between models.

**Key Inputs**:
- `full_model_dir`: Full model results directory
- `segmented_model_dir`: Segmented model results directory
- `output_dir`: Directory to save analysis

**Key Outputs**:
- KS test results
- Probability distribution visualizations
- Performance metrics comparison
- Confidence calibration analysis

## mcnemar_test.py
**Purpose**: Performs McNemar's test to determine if there is a statistically significant difference between models.

**Key Inputs**:
- `full_model_dir`: Full model results directory
- `segment_model_dir`: Segmented model results directory
- `output_dir`: Directory to save comparison results

**Key Outputs**:
- McNemar's test statistics
- Contingency table visualization
- Per-family statistical analysis
- Comprehensive model comparison report

## mcnemar_family_comparison.py
**Purpose**: Provides detailed analysis of per-family performance differences using McNemar's test results.

**Key Inputs**:
- `results_dir`: Directory containing comparison results
- `output_dir`: Directory to save analysis results

**Key Outputs**:
- Family-wise performance analysis
- Statistical significance visualizations
- Sample count vs. accuracy difference plots
- p-value heatmaps by family