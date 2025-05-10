"""
Components for visualizing model metrics.
"""

import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc


def create_metrics_controls():
    """
    Create control elements for metrics visualization.

    Returns:
        tuple: (model_selector, metric_dropdown) Dash components
    """
    # Model selector radio buttons
    model_selector = dcc.RadioItems(
        id='model-selector',
        options=[
            {'label': 'Full Image Model', 'value': 'Full Image'},
            {'label': 'Segment Model', 'value': 'Segment Model'}
        ],
        value='Full Image',
        inline=True,
        className="radio-group",
        inputStyle={"marginRight": "0.35rem"}
        # spacing between button and label
    )

    # Metric dropdown
    metric_dropdown = dcc.Dropdown(
        id='metric-dropdown',
        options=[{'label': m, 'value': m} for m in
                 ['Precision', 'Recall', 'F1', 'Accuracy', 'Weighted F1']],
        value='Precision',
        clearable=False
    )

    return model_selector, metric_dropdown


def create_metrics_plot():
    """
    Create metrics plot container with controls.

    Returns:
        html.Div: Complete metrics plot component
    """
    # Get control components
    model_selector, metric_dropdown = create_metrics_controls()

    # Create the complete component
    return html.Div([
        html.Div([
            html.Div("Select Model:", className="plot-label",
                     style={'marginBottom': '0.25rem'}),
            model_selector
        ], style={'marginBottom': '1rem'}),

        html.Div([
            html.Div("Select Metric Type:", className="plot-label",
                     style={'marginBottom': '0.25rem'}),
            metric_dropdown
        ], style={'marginBottom': '0.5rem'}),

        dcc.Graph(id='bar-chart', style={"width": "100%", "height": "400px",
                                         "marginBottom": "0"}),
        html.Div(id='std-dev-display', className='metric-summary'),
    ])


def create_overall_metric_chart(filtered_data, selected_metric, selected_model):
    """
    Create a bar chart for overall metrics (Accuracy, Weighted F1).

    Args:
        filtered_data (DataFrame): Filtered metrics data
        selected_metric (str): Selected metric type
        selected_model (str): Selected model type

    Returns:
        tuple: (figure, summary_text) Plotly figure and summary text
    """
    # Create bar chart
    fig = px.bar(
        filtered_data,
        x='Fold',
        y='Score',
        color='BaseMetric',
        title=f"{selected_metric} Across Folds ({selected_model})",
        labels={'Score': 'Score', 'Fold': 'Fold'},
        text_auto='.3f',
        color_discrete_sequence=['gray']
    )

    # Calculate summary statistics
    mean = filtered_data['Score'].mean()
    std = filtered_data['Score'].std()
    std_text = f"Average: {mean:.4f} | Standard Deviation: {std:.4f}"

    return fig, std_text


def create_class_metric_chart(filtered_data, selected_metric, selected_model):
    """
    Create a grouped bar chart for class-specific metrics.

    Args:
        filtered_data (DataFrame): Filtered metrics data
        selected_metric (str): Selected metric type
        selected_model (str): Selected model type

    Returns:
        tuple: (figure, summary_text) Plotly figure and summary text
    """
    # Split data by class
    class0 = filtered_data[filtered_data['Class'] == 'Class 0']
    class1 = filtered_data[filtered_data['Class'] == 'Class 1']

    # Create grouped bar chart
    fig = go.Figure([
        go.Bar(name='Class 0', x=class0['Fold'], y=class0['Score'],
               marker=dict(color='#3182bd')),
        go.Bar(name='Class 1', x=class1['Fold'], y=class1['Score'],
               marker=dict(color='#08519c'))
    ])

    # Update layout
    fig.update_layout(
        barmode='group',
        title=f"{selected_metric} Across Folds ({selected_model})",
        yaxis=dict(range=[0.8, 1.0]),
        xaxis_title='Fold',
        yaxis_title='Score'
    )

    # Add model metadata annotation
    add_model_metadata(fig, selected_model)

    # Calculate summary statistics for both classes
    c0_mean = class0['Score'].mean()
    c0_std = class0['Score'].std()
    c1_mean = class1['Score'].mean()
    c1_std = class1['Score'].std()

    std_text = (
        f"Average (Class 0): {c0_mean:.4f} | Standard Deviation (Class 0): {c0_std:.4f} | "
        f"Average (Class 1): {c1_mean:.4f} | Standard Deviation (Class 1): {c1_std:.4f}"
    )

    return fig, std_text


def add_model_metadata(fig, selected_model):
    """
    Add model metadata annotation to a figure.

    Args:
        fig (go.Figure): Plotly figure
        selected_model (str): Selected model type

    Returns:
        None: Modifies figure in-place
    """
    # Determine metadata based on model type
    if selected_model == 'Full Image':
        metadata_lines = [
            "Image Size: 256", "Batch Size: 32", "Epochs: 50",
            "Train: 9488", "Val: 2034", "Test: 2881"
        ]
    else:
        metadata_lines = [
            "Image Size: 128", "Batch Size: 32", "Epochs: 30",
            "Train: 9488", "Val: 2034", "Test: 2881"
        ]

    metadata_text = "<br>".join(metadata_lines)

    # Add annotation to figure
    fig.add_annotation(
        text=metadata_text,
        xref="paper", yref="paper",
        x=1.05,  # push it right, just beyond the legend
        y=0.5,  # center vertically next to the plot
        showarrow=False,
        font=dict(size=10, color="#666"),
        align="left",
        textangle=0,
        valign="middle",
        xanchor="left",
        yanchor="middle"
    )