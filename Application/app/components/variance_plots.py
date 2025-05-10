"""
Components for visualizing variance data.
"""

import plotly.express as px
import plotly.figure_factory as ff
from dash import html, dcc
import dash_bootstrap_components as dbc  # Correct import for dbc

# Color palette for segment graphs
SEGMENT_COLORS = ['#084594', '#2171b5', '#6baed6', '#c6dbef']


def create_family_variance_heatmap(family_variances):
    """
    Create a heatmap showing average variance by family and segment.

    Args:
        family_variances (DataFrame): DataFrame with family-wise segment variances

    Returns:
        dcc.Graph: Plotly heatmap of family variances
    """
    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=family_variances.values,
        x=list(family_variances.columns),
        y=family_variances.index.tolist(),
        colorscale='Blues',
        showscale=True
    )

    # Update layout
    fig.update_layout(
        title=dict(text='Avg Segment Variance by Family', y=0.99),
        margin=dict(l=10, r=10, t=50, b=10),
        height=400
    )

    return dcc.Graph(figure=fig)


def create_segment_selection_barchart(percent_df):
    """
    Create a stacked bar chart showing segment selection frequency by family.

    Args:
        percent_df (DataFrame): DataFrame with segment selection percentages

    Returns:
        dcc.Graph: Plotly bar chart of segment selection frequencies
    """
    # Melt dataframe for Plotly Express
    melted_df = percent_df.reset_index().melt(
        id_vars='family',
        var_name='Segment',
        value_name='Percentage'
    )

    # Create stacked bar chart
    fig = px.bar(
        melted_df,
        x='family',
        y='Percentage',
        color='Segment',
        barmode='stack',
        title='Segment Selection Frequency by Family (%)',
        category_orders={'Segment': ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']},
        height=400,
        color_discrete_sequence=SEGMENT_COLORS
    )

    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor='left', x=1.02),
        bargap=0,
        title=dict(y=0.99)
    )

    return dcc.Graph(figure=fig)


def create_family_variance_plots(family_variances, percent_df):
    """
    Create a row with family variance heatmap and segment selection bar chart.

    Args:
        family_variances (DataFrame): DataFrame with family-wise segment variances
        percent_df (DataFrame): DataFrame with segment selection percentages

    Returns:
        dbc.Row: Row containing both plots
    """
    # Create both plots
    bar_chart = create_segment_selection_barchart(percent_df)
    heatmap = create_family_variance_heatmap(family_variances)

    # Return as a row with two columns
    return dbc.Row([
        dbc.Col(bar_chart, md=6),
        dbc.Col(heatmap, md=6),
    ], className="mb-4")


def create_variance_sample_tile(base_path, overlay_path):
    """
    Create a tile showing base image with variance overlay.

    Args:
        base_path (str): Path to base image
        overlay_path (str): Path to overlay image

    Returns:
        html.Div: Tile component
    """
    return html.Div([
        html.Div([
            html.Img(src=f"/{base_path}", className="base"),
            html.Img(src=f"/{overlay_path}", className="overlay")
        ], className="tile-inner")
    ], className="tile")


def create_variance_sample_grid(tiles):
    """
    Create a grid of variance sample tiles.

    Args:
        tiles (list): List of tile components

    Returns:
        html.Div: Grid component
    """
    return html.Div(tiles, className="sample-grid")


def create_variance_family_dropdown(variance_families):
    """
    Create a dropdown for selecting variance family.

    Args:
        variance_families (list): List of family names

    Returns:
        dcc.Dropdown: Family selection dropdown
    """
    options = [{"label": fam, "value": fam} for fam in variance_families]

    return dcc.Dropdown(
        id="variance-family-dd",
        options=options,
        value=options[0]['value'] if options else None,
        clearable=False,
        style={"width": "300px", "marginBottom": "0.5rem"}
    )


def create_variance_overlay_section(variance_families):
    """
    Create collapsible section for variance overlay samples.

    Args:
        variance_families (list): List of family names

    Returns:
        html.Div: Complete variance overlay section
    """
    return html.Div([
        dbc.Button(
            "View Variance Heatmap Overlay Samples",
            id="variance-collapse-toggle",
            className="mb-2",
            color="secondary",
            n_clicks=0
        ),
        dbc.Collapse(
            html.Div([
                html.Div("Select Family:", className="plot-label",
                         style={'marginBottom': '0.5rem'}),
                create_variance_family_dropdown(variance_families),
                html.Div(
                    id="variance-description",
                    style={
                        "fontSize": "0.85rem",
                        "fontStyle": "italic",
                        "color": "#666",
                        "marginBottom": "1.5rem",
                        "textAlign": "center"
                    }
                ),
                html.Div(id="variance-sample-grid", className="sample-grid")
            ]),
            id="variance-collapse-container",
            is_open=False
        )
    ], className="mb-4")