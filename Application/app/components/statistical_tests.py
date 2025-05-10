"""
Components for visualizing statistical test results.
"""

import numpy as np
import plotly.figure_factory as ff
from dash import html, dcc
import dash_bootstrap_components as dbc  # Correct import for dbc


def create_mcnemar_plot(mcnemar_data):
    """
    Create a visualization of McNemar's test results.

    Args:
        mcnemar_data (dict): Dictionary with McNemar's test data

    Returns:
        tuple: (figure_component, summary_component)
    """
    # Reorder the matrix to account for Plotly's rendering
    # Plotly renders the first row at the bottom, so we need to flip the order
    matrix = np.array([
        # Bottom row in the visual (appears at top in the matrix)
        [mcnemar_data['full_wrong_segment_correct'],
         mcnemar_data['both_wrong']],
        # Top row in the visual (appears at bottom in the matrix)
        [mcnemar_data['both_correct'],
         mcnemar_data['full_correct_segment_wrong']]
    ])

    # Also reorder the text labels to match
    text_labels = [
        [f"c\n{mcnemar_data['full_wrong_segment_correct']}",
         f"d\n{mcnemar_data['both_wrong']}"],
        [f"a\n{mcnemar_data['both_correct']}",
         f"b\n{mcnemar_data['full_correct_segment_wrong']}"]
    ]

    # Create the heatmap with correct ordering of axis labels
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=['Correct', 'Wrong'],  # Full Image Model
        y=['Wrong', 'Correct'],  # Segment Model - REVERSED ORDER!
        annotation_text=text_labels,
        colorscale='Blues',
        showscale=True
    )

    # Update layout
    fig.update_layout(
        title="McNemar Contingency Matrix",
        xaxis_title="Full Image Model",
        yaxis_title="Segment Model",
        margin=dict(l=40, r=40, t=80, b=40),
        height=350,
        xaxis=dict(side='bottom')
    )

    # Generate summary text with test results
    p_value = mcnemar_data['mcnemar_p_value']
    summary_text = f"""
    **Statistical Test Result**
    McNemar's Test Statistic: {mcnemar_data['mcnemar_statistic']:.3f}  
    p-value: {p_value:.2e}  
    """

    # Return components
    figure_component = dcc.Graph(figure=fig, style={"marginBottom": "1rem"})
    summary_component = dcc.Markdown(
        summary_text,
        className="metric-summary",
        style={"textAlign": "right"}
    )

    return figure_component, summary_component


def create_mcnemar_section(mcnemar_data, mcnemar_text):
    """
    Create a complete section for McNemar's test visualization.

    Args:
        mcnemar_data (dict): Dictionary with McNemar's test data
        mcnemar_text (dict): Dictionary with text content

    Returns:
        html.Div: Complete section component
    """
    # Create McNemar plot components
    mcnemar_fig, mcnemar_summary = create_mcnemar_plot(mcnemar_data)

    # Create complete section
    return html.Div([
        # Section heading
        html.Hr(),
        html.H5("McNemar Testing", className="heading"),
        html.Hr(),

        # Explanatory text
        html.Div(
            mcnemar_text.mcnemar_paragraph_1,
            className="paragraph left-align",
            **{"data-aos": "fade-up"}
        ),

        # Formula
        html.Div([
            dcc.Markdown(
                r'''
                $$
                \chi^2 = \frac{(b-c)^2}{b+c}
                $$
                ''',
                mathjax=True,  # Explicitly enable MathJax
                style={"textAlign": "center", "margin": "20px 0"}
            )
        ]),

        # Formula explanation
        html.Div([
            dcc.Markdown(
                r'''
                where:

                b represents cases where the first model succeeded but the second failed (59 samples)

                c represents cases where the first model failed but the second succeeded (16 samples)
                ''',
                mathjax=True,
                style={"textAlign": "center", "fontSize": "0.9rem",
                       "margin": "0 0 20px 0"}
            )
        ]),

        # Additional explanatory text
        html.Div(
            mcnemar_text.mcnemar_paragraph_2,
            className="paragraph left-align",
            **{"data-aos": "fade-up"}
        ),

        # Results visualization
        html.Div([
            mcnemar_fig,
            mcnemar_summary
        ], style={"maxWidth": "800px", "margin": "auto"})
    ])


def create_ks_test_section(benign_prob_dist_img, malware_prob_dist_img, ks_text):
    """
    Create a complete section for Kolmogorov-Smirnov test visualization.

    Args:
        benign_prob_dist_img (str): Path to benign distribution image
        malware_prob_dist_img (str): Path to malware distribution image
        ks_text (dict): Dictionary with text content

    Returns:
        html.Div: Complete section component
    """
    return html.Div([
        # Section heading
        html.Hr(),
        html.H5("Kolmogorov-Smirnov Test", className="heading"),
        html.Hr(),

        # First paragraph
        html.Div(
            ks_text.ks_paragraph_1,
            className="paragraph left-align",
            **{"data-aos": "fade-up"}
        ),

        # LaTeX formula
        html.Div([
            dcc.Markdown(
                r'''
                $$
                D = \max_x |F_1(x) - F_2(x)|
                $$
                ''',
                mathjax=True,
                style={"textAlign": "center", "margin": "20px 0"}
            )
        ]),

        # Formula explanation
        html.Div([
            dcc.Markdown(
                r'''
                Where $F_1(x)$ is the empirical CDF of the first sample and $F_2(x)$ is the empirical CDF of the second sample
                ''',
                mathjax=True,
                style={"textAlign": "center", "fontSize": "0.9rem",
                       "margin": "0 0 20px 0"}
            )
        ]),

        # Second paragraph
        html.Div(
            ks_text.ks_paragraph_2,
            className="paragraph left-align",
            **{"data-aos": "fade-up"}
        ),

        # Bullet points for benign observations
        html.Div([
            html.Ul([
                html.Li(
                    ks_text.ks_benign_bullet1,
                    style={"textAlign": "left"}
                ),
                html.Li(
                    ks_text.ks_benign_bullet2,
                    style={"textAlign": "left"}
                ),
            ], style={"marginLeft": "2rem", "marginBottom": "1.5rem"})
        ], className="paragraph", **{"data-aos": "fade-up"}),

        # Third paragraph
        html.Div(
            ks_text.ks_paragraph_3,
            className="paragraph left-align",
            **{"data-aos": "fade-up"}
        ),

        # Distribution images
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6(
                        "Benign Sample Distribution",
                        style={"textAlign": "center", "marginBottom": "10px"}
                    ),
                    html.Img(
                        src=benign_prob_dist_img,
                        style={'width': '100%', 'height': '500px'},
                        alt="Benign probability distribution comparison"
                    )
                ], xs=12, md=6),
                dbc.Col([
                    html.H6(
                        "Malware Sample Distribution",
                        style={"textAlign": "center", "marginBottom": "10px"}
                    ),
                    html.Img(
                        src=malware_prob_dist_img,
                        style={'width': '100%', 'height': '500px'},
                        alt="Malware probability distribution comparison"
                    )
                ], xs=12, md=6),
            ], className="mb-4")
        ])
    ])


def create_calibration_section(calibration_curves_img, calibration_text):
    """
    Create a section for model calibration visualization.

    Args:
        calibration_curves_img (str): Path to calibration curves image
        calibration_text (str): Explanatory text

    Returns:
        html.Div: Complete section component
    """
    return html.Div([
        # Section heading
        html.Hr(),
        html.H5("Model Calibration", className="heading"),
        html.Hr(),

        # Explanatory text
        html.Div(calibration_text,
                 className="paragraph left-align",
                 **{"data-aos": "fade-up"}),

        # Calibration curves image
        html.Div([
            html.Img(
                src=calibration_curves_img,
                style={'width': '100%', 'height': '500px'},
                alt="Calibration curves comparison")
        ])
    ])