"""
Data processing utilities for analytics page.
Contains functions for transforming data for visualizations.
"""

import pandas as pd
import numpy as np


def prepare_metrics_data(cv_metrics):
    """
    Transform cross-validation metrics dataframe to long format for visualization.

    Args:
        cv_metrics (DataFrame): DataFrame containing model metrics

    Returns:
        DataFrame: Processed dataframe in long format
    """
    # Melt data for class-specific metrics
    long_df = pd.melt(
        cv_metrics,
        id_vars=['Model', 'Fold'],
        value_vars=[
            'Class 0 Precision', 'Class 0 Recall', 'Class 0 F1',
            'Class 1 Precision', 'Class 1 Recall', 'Class 1 F1'
        ],
        var_name='Metric',
        value_name='Score'
    )

    # Extract class and base metric information
    long_df['Class'] = long_df['Metric'].str.extract(r'(Class \d)')
    long_df['BaseMetric'] = long_df['Metric'].str.extract(
        r'(Precision|Recall|F1)')

    # Add overall metrics (not class-specific)
    extra_rows = cv_metrics.melt(
        id_vars=['Model', 'Fold'],
        value_vars=['Accuracy', 'Weighted F1'],
        var_name='BaseMetric',
        value_name='Score'
    )
    extra_rows['Class'] = 'Overall'

    # Combine all metrics
    return pd.concat(
        [long_df[['Model', 'Fold', 'BaseMetric', 'Score', 'Class']],
         extra_rows],
        ignore_index=True
    )


def prepare_variance_data(variance_df):
    """
    Rename variance columns for better readability.

    Args:
        variance_df (DataFrame): Raw variance dataframe

    Returns:
        DataFrame: Processed variance dataframe
    """
    # Create a copy to avoid modifying the original
    processed_df = variance_df.copy()

    # Rename columns for better readability
    processed_df.rename(columns={
        'segment_0_variance': 'Top-Left',
        'segment_1_variance': 'Top-Right',
        'segment_2_variance': 'Bottom-Left',
        'segment_3_variance': 'Bottom-Right'
    }, inplace=True)

    return processed_df


def get_top_families_variance(variance_df, top_n=10):
    """
    Get variance statistics for top N families.

    Args:
        variance_df (DataFrame): Variance dataframe
        top_n (int): Number of top families to include

    Returns:
        tuple: (top_families list, family_variances DataFrame)
    """
    # Get the top N families by count
    top_families = variance_df['family'].value_counts().nlargest(
        top_n).index.tolist()

    # Calculate mean variance per segment for each family
    family_variances = \
    variance_df[variance_df['family'].isin(top_families)].groupby('family')[
        ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    ].mean().round(0).astype(int)

    return top_families, family_variances


def get_segment_selection_counts(variance_df, top_families):
    """
    Calculate segment selection frequency by family.

    Args:
        variance_df (DataFrame): Variance dataframe
        top_families (list): List of family names to include

    Returns:
        DataFrame: Percentage of segment selection by family
    """
    # Count selections per segment per family
    count_df = variance_df[variance_df['family'].isin(top_families)].groupby(
        ['family', 'selected_segment']
    ).size().unstack(fill_value=0)

    # Rename columns for consistency
    count_df.columns = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']

    # Convert to percentages
    percent_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

    return percent_df