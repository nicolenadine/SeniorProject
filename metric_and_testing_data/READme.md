# Model Evaluation and Analysis Files

This directory contains a collection of files related to the evaluation and analysis of machine learning models, likely comparing full-image vs. segmented approaches to malware detection.

## JSON Files

- **ks_test_results.json**: Statistical analysis results from Kolmogorov-Smirnov tests, comparing distributions between benign and malware samples.

- **mcnemar_comparison_results.json**: Detailed McNemar's test results comparing two models (full-image vs. segment-based), including contingency tables, performance metrics, and family-level analysis.

- **seg1_ensemble_metrics.json**: Performance metrics for a segment-based ensemble model, including accuracy, precision, recall, F1 score, and confusion matrix.

## CSV Files

- **cross_validation_metrics.csv**: Results from cross-validation experiments, with metrics for multiple model folds and classes.

- **per_sample_metrics.csv**: Detailed per-sample performance analysis comparing full-image and segment-based predictions, including probabilities and log loss.

- **prediction_differences.csv**: Focused dataset highlighting samples where the full and segmented models made different predictions.

- **sample_level_predictions_rounded.csv**: Sample-level prediction results, including probabilities and selected segments.

- **segment_variance_data_cleaned.csv**: Analysis of variance across different segments, with information about which segment was selected.

