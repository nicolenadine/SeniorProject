"""
Analytics page layout for the Dash application.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd

# Import text content
from content import analytics_text
from content.data_sources import (
    cv_metrics, conf_data, variance_df, prediction_df,
    calibration_curves_img, benign_prob_dist_img, malware_prob_dist_img,
    mcnemar_data
)

# Import components
from components.confusion_matrix import create_confusion_matrix
from components.metrics_plots import create_metrics_plot
from components.variance_plots import (
    create_family_variance_plots, create_variance_overlay_section
)
from components.data_tables import create_collapsible_tables_section
from components.statistical_tests import (
    create_mcnemar_section, create_ks_test_section, create_calibration_section
)

# Import utilities
from utils.data_processing import (
    prepare_metrics_data, prepare_variance_data,
    get_top_families_variance, get_segment_selection_counts
)

# Register the page
dash.register_page(__name__, path="/analytics", name="Analytics")

# --------------- LOAD AND PROCESS DATA ------------

# Load variance metadata
variance_meta = pd.read_parquet("assets/random_gradcam/random_meta.parquet")
variance_families = sorted(variance_meta['family'].unique())

# Process metrics data
long_df = prepare_metrics_data(cv_metrics)

# Process variance data
processed_variance_df = prepare_variance_data(variance_df)

# Drop file_path column from predictions for display
display_prediction_df = prediction_df.drop(columns=['file_path'])

# Process data for family variance analysis
top_families, family_variances = get_top_families_variance(processed_variance_df)
segment_selection_percents = get_segment_selection_counts(processed_variance_df, top_families)

# ------------- CREATE PAGE COMPONENTS -------------

# Create confusion matrix component
confusion_matrix_component = create_confusion_matrix(conf_data)

# Create family variance plots
family_variance_plots = create_family_variance_plots(family_variances, segment_selection_percents)

# Create variance overlay section
variance_overlay_section = create_variance_overlay_section(variance_families)

# Create tables section
tables_section = create_collapsible_tables_section(processed_variance_df, display_prediction_df)

# Create metrics plot
metrics_plot = create_metrics_plot()

# Create statistical test sections
mcnemar_section = create_mcnemar_section(mcnemar_data, analytics_text)
ks_section = create_ks_test_section(benign_prob_dist_img, malware_prob_dist_img, analytics_text)
calibration_section = create_calibration_section(
    calibration_curves_img, analytics_text.calibration_curve_paragraph
)

# --------------- PAGE LAYOUT ------------------
layout = html.Div([
    # Segmentation Approach Section
    html.H5("Segmentation Approach", className="heading"),
    html.Div(analytics_text.initial_segment_fail_paragraph,
             className="paragraph left-align"),
    confusion_matrix_component,

    html.Hr(),

    # New Segment Paragraph
    dbc.Row([
        html.Div(analytics_text.new_segment_paragraph,
                 className="paragraph left-align"),
    ], className="mb-4"),

    # Family-Level Analysis Section
    html.H5("Family-Level Segment Analysis", className="heading"),
    dbc.Row([
        html.Div(analytics_text.segment_analysis_paragraph,
                 className="paragraph left-align"),
    ], className="mb-4"),
    html.Div([family_variance_plots]),
    variance_overlay_section,
    tables_section,

    # Cross-Validation Section
    html.Hr(),
    html.H5("Cross-Validation", className="heading"),
    html.Hr(),
    dbc.Row([
        dbc.Col(html.Div(analytics_text.cross_validation_paragraph,
                         className="paragraph left-align"), xs=12, md=6),
        dbc.Col(html.Div(metrics_plot),
                xs=12, md=6),
    ], className="mb-2", style={"padding-bottom": "25px"}),

    # Calibration Section
    calibration_section,

    # McNemar Testing Section
    mcnemar_section,

    # KS Test Section
    ks_section

], style={"maxWidth": "1100px", "margin": "auto"})


def init_callbacks(app):
    """
    Initialize all callbacks for the analytics page.

    Args:
        app: Dash application instance
    """
    # Import callback registration function here to avoid circular imports
    from callbacks.analytics_callbacks import register_analytics_callbacks

    # Register the callbacks
    register_analytics_callbacks(
        app,
        long_df,
        processed_variance_df,
        display_prediction_df,
        variance_meta
    )