"""
Confusion matrix component for visualizing model performance.
"""

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc


def create_confusion_matrix(conf_data):
    """
    Create a confusion matrix visualization with associated metrics.

    Args:
        conf_data (dict): Dictionary containing confusion matrix and metrics

    Returns:
        html.Div: Dash component containing the confusion matrix and metrics
    """
    # Extract confusion matrix data
    conf_matrix = conf_data["confusion_matrix"]

    # Convert to numpy for formatting
    z = np.array(conf_matrix)
    x_labels = ['Benign', 'Malware']
    y_labels = ['Benign', 'Malware']

    # Create the confusion matrix figure
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale='Blues',
            showscale=True,
            text=z,
            texttemplate="%{text}",
            hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count: %{z}<extra></extra>"
        )
    )

    # Update layout
    fig.update_layout(
        title="Confusion Matrix: Failed Segmentation (Majority Voting)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        margin=dict(l=50, r=50, t=40, b=40),
        height=350
    )

    # Create the summary text with metrics
    summary_text = html.Div([
        html.Span(
            f"Accuracy: {conf_data['accuracy']:.4f} | Recall: {conf_data['recall']:.4f} | F1 Score: {conf_data['f1_score']:.4f}"
        ),
        html.Br(),
        html.Span(
            f"Avg Precision: {conf_data['avg_precision']:.4f} | Voting Method: {conf_data['voting_method'].capitalize()} | Voting Threshold: {conf_data['voting_threshold']}"
        )
    ], className="confusion-summary")

    # Return the complete component
    return html.Div([
        dcc.Graph(figure=fig, style={"width": "100%"}),
        summary_text
    ], style={"maxWidth": "800px", "margin": "auto"}, **{"data-aos": "zoom-in"})