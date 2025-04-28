import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import dash_table

from content import analytics_text
from content.data_sources import *


dash.register_page(__name__, path="/analytics", name="Analytics")

# --------------- LOAD METADATA ------------
variance_meta = pd.read_parquet("assets/random_gradcam/random_meta.parquet")
variance_families = sorted(variance_meta['family'].unique())

# ---  Global variance heatmap overlay dropdown options ---
variance_family_options = [
    {"label": fam, "value": fam} for fam in variance_families
]

# ----------- TOP RIGHT BAR PLOT ----------------

# Melt data
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

long_df['Class'] = long_df['Metric'].str.extract(r'(Class \d)')
long_df['BaseMetric'] = long_df['Metric'].str.extract(r'(Precision|Recall|F1)')

extra_rows = cv_metrics.melt(id_vars=['Model', 'Fold'], value_vars=['Accuracy', 'Weighted F1'], var_name='BaseMetric', value_name='Score')
extra_rows['Class'] = 'Overall'
long_df = pd.concat([long_df[['Model', 'Fold', 'BaseMetric', 'Score', 'Class']], extra_rows], ignore_index=True)

# Controls
model_selector = dcc.RadioItems(
    id='model-selector',
    options=[
        {'label': 'Full Image Model', 'value': 'Full Image'},
        {'label': 'Segment Model', 'value': 'Segment Model'}
    ],
    value='Full Image',
    inline=True,
    className="radio-group",
    inputStyle={"marginRight": "0.35rem"}  # spacing between button and label
)


metric_dropdown = dcc.Dropdown(
    id='metric-dropdown',
    options=[{'label': m, 'value': m} for m in ['Precision', 'Recall', 'F1', 'Accuracy', 'Weighted F1']],
    value='Precision',
    clearable=False
)

right_plot_1 = html.Div([
    html.Div([
        html.Div("Select Model:", className="plot-label", style={
            'marginBottom': '0.25rem'}),
        model_selector
    ], style={'marginBottom': '1rem'}),

    html.Div([
        html.Div("Select Metric Type:", className="plot-label", style={
            'marginBottom': '0.25rem'}),
        metric_dropdown
    ], style={'marginBottom': '0.5rem'}),

    dcc.Graph(id='bar-chart', style={"width": "100%", "height": "400px",
                                     "marginBottom": "0"}),
    html.Div(id='std-dev-display', className='metric-summary'),

])

# ---------- CONFUSION MATRIX --------------
# Load confusion matrix data
conf_matrix = conf_data["confusion_matrix"]

# Convert to numpy for easy formatting
z = np.array(conf_matrix)
x_labels = ['Benign', 'Malware']
y_labels = ['Benign', 'Malware']

conf_matrix = dcc.Graph(
    figure=go.Figure(
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
    ).update_layout(
        title="Confusion Matrix: Failed Segmentation (Majority Voting)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        margin=dict(l=50, r=50, t=40, b=40),
        height=350
    ),
    style={"width": "100%"}
)

# ------ TABLES ----------------

# Rename column names for variance
variance_df.rename(columns={
    'segment_0_variance': 'Top-Left',
    'segment_1_variance': 'Top-Right',
    'segment_2_variance': 'Bottom-Left',
    'segment_3_variance': 'Bottom-Right'
}, inplace=True)

# Drop file_path column from predictions
display_prediction_df = prediction_df.drop(columns=['file_path'])


# ----------- FAMILY VARIANCE & SEGMENT SELECTION PLOTS -------------
def create_family_variance_and_selection_plots():

    top_families = variance_df['family'].value_counts().nlargest(10).index.tolist()

    # Family-wise average variance heatmap
    family_variances = variance_df[variance_df['family'].isin(top_families)].groupby('family')[
        ['Top-Left', 'Top-Right', 'Bottom-Left',
         'Bottom-Right']
    ].mean().round(0).astype(int)
    family_variances.columns = ['Top-Left', 'Top-Right', 'Bottom-Left',
                                'Bottom-Right']

    fig_heatmap = ff.create_annotated_heatmap(
        z=family_variances.values,
        x=list(family_variances.columns),
        y=family_variances.index.tolist(),
        colorscale='Blues',
        showscale=True
    )
    fig_heatmap.update_layout(
        title=dict(text='Avg Segment Variance by Family', y=0.99),
        margin=dict(l=10, r=10, t=50, b=10),
        height=400
    )

    # Segment selection bar chart
    count_df = variance_df[variance_df['family'].isin(top_families)].groupby(['family', 'selected_segment']).size().unstack(fill_value=0)
    count_df.columns = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    percent_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

    fig_bar = px.bar(
        percent_df.reset_index().melt(id_vars='family', var_name='Segment', value_name='Percentage'),
        x='family', y='Percentage', color='Segment', barmode='stack',
        title='Segment Selection Frequency by Family (%)',
        category_orders={'Segment': ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']},
        height=400,
        color_discrete_sequence=['#084594', '#2171b5', '#6baed6', '#c6dbef']
    )
    fig_bar.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor='left', x=1.02),
        bargap=0,
        title=dict(y=0.99)
    )

    return dcc.Graph(figure=fig_bar), dcc.Graph(figure=fig_heatmap)


def load_mcnemar_summary():

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

    metrics = mcnemar_data['metrics']
    better_model = mcnemar_data['better_model']
    p_value = mcnemar_data['mcnemar_p_value']

    summary_text = f"""
    **Statistical Test Result**
    McNemar's Test Statistic: {mcnemar_data['mcnemar_statistic']:.3f}  
    p-value: {p_value:.2e}  
    """

    return dcc.Graph(figure=fig), dcc.Markdown(summary_text)


# ------- LAYOUT ------------------
family_bar_graph, family_heatmap_graph = create_family_variance_and_selection_plots()
mcnemar_fig, mcnemar_summary = load_mcnemar_summary()

layout = html.Div([
    html.H5("Segmentation Approach", className="heading"),
    html.Div(analytics_text.initial_segment_fail_paragraph,
             className="paragraph left-align", **{"data-aos": "fade-up"}),
    html.Div([
        conf_matrix,
        html.Div([
            html.Span(
                f"Accuracy: {conf_data['accuracy']:.4f} | Recall: {conf_data['recall']:.4f} | F1 Score: {conf_data['f1_score']:.4f}"
            ),
            html.Br(),
            html.Span(
                f"Avg Precision: {conf_data['avg_precision']:.4f} | Voting Method: {conf_data['voting_method'].capitalize()} | Voting Threshold: {conf_data['voting_threshold']}"
            )
        ], className="confusion-summary")
    ], style={"maxWidth": "800px", "margin": "auto"},
        **{"data-aos": "zoom-in"}),

    html.Hr(),

    dbc.Row([
        html.Div(analytics_text.new_segment_paragraph, className="paragraph "
                                                                 "left-align",
                 **{"data-aos": "fade-up"}),
    ], className="mb-4"),


    html.H5("Family-Level Segment Analysis", className="heading"),

    dbc.Row([
        html.Div(analytics_text.segment_analysis_paragraph,
                 className="paragraph left-align", **{"data-aos": "fade-up"}),
    ], className="mb-4"),

    html.Div([
        dbc.Row([
            dbc.Col(family_bar_graph, md=6),
            dbc.Col(family_heatmap_graph, md=6),
        ], className="mb-4")
    ]),

    html.Div([
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

                dcc.Dropdown(
                    id="variance-family-dd",
                    options=variance_family_options,
                    value=variance_family_options[0]['value'],
                    clearable=False,
                    style={"width": "300px", "marginBottom": "0.5rem"}
                ),

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
    ], className="mb-4"),


    html.Div([
        dbc.Button(
            "View Table Data",
            id="collapse-toggle",
            className="mb-2",
            color="secondary",
            n_clicks=0
        ),
        dbc.Collapse(
            html.Div([
                html.Div("Select Table to View:", className="plot-label",
                         style={'marginBottom': '0.5rem'}),
                dcc.RadioItems(
                    id='table-toggle',
                    options=[
                        {'label': 'Segment Variance', 'value': 'variance'},
                        {'label': 'Test Predictions', 'value': 'predictions'}
                    ],
                    value='variance',
                    inline=True,
                    className="radio-group",
                    inputStyle={"marginRight": "0.35rem"}
                ),
                html.Div(id='table-container')
            ]),
            id="collapse-container",
            is_open=False
        )

    ], className="mb-4"),

    html.Hr(),
    html.H5(" Cross-Validation", className="heading"),
    html.Hr(),

    dbc.Row([
        dbc.Col(html.Div(analytics_text.cross_validation_paragraph,
                         className="paragraph left-align",
                         **{"data-aos": "fade-right"}), xs=12, md=6),
        dbc.Col(html.Div(right_plot_1, **{"data-aos": "fade-left"}), xs=12,
                md=6),
    ], className="mb-2", style={"padding-bottom": "25px"}),

    html.Hr(),
    html.H5("Model Calibration", className="heading"),
    html.Hr(),

    html.Div(analytics_text.calibration_curve_paragraph,
             className="paragraph left-align",
             **{"data-aos": "fade-up"}),

    html.Div([
        html.Img(
            src=calibration_curves_img,
            style={'width': '100%', 'height': '500px'},
            alt="Benign probability distribution comparison")
        ]),

    # Replace everything from the McNemar test section to the end of your layout

    html.Hr(),
    html.H5("McNemar Testing", className="heading"),
    html.Hr(),

    html.Div(
        analytics_text.mcnemar_paragraph_1,
        className="paragraph left-align",
        **{"data-aos": "fade-up"}
    ),

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

    html.Div(
        analytics_text.mcnemar_paragraph_2,
        className="paragraph left-align",
        **{"data-aos": "fade-up"}
    ),

    html.Div([
        html.Div(mcnemar_fig, style={"marginBottom": "1rem"}),

        # Right-align the summary
        html.Div(
            mcnemar_summary,
            className="metric-summary",
            style={"textAlign": "right"}
        )
    ], style={"maxWidth": "800px", "margin": "auto"}),

    # KS Test Section
    html.Hr(),
    html.H5("Kolmogorov-Smirnov Test", className="heading"),
    html.Hr(),

    # First paragraph
    html.Div(
        analytics_text.ks_paragraph_1,
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
        analytics_text.ks_paragraph_2,
        className="paragraph left-align",
        **{"data-aos": "fade-up"}
    ),

    # Bullet points for benign observations (text NOT centered within bullets)
    html.Div([
        html.Ul([
            html.Li(
                analytics_text.ks_benign_bullet1,
                style={"textAlign": "left"}
            ),
            html.Li(
                analytics_text.ks_benign_bullet2,
                style={"textAlign": "left"}
            ),
        ], style={"marginLeft": "2rem", "marginBottom": "1.5rem"})
    ], className="paragraph", **{"data-aos": "fade-up"}),

    # Third paragraph
    html.Div(
        analytics_text.ks_paragraph_3,
        className="paragraph left-align",
        **{"data-aos": "fade-up"}
    ),

    # Images
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
], style={"maxWidth": "1100px", "margin": "auto"})



def update_chart(selected_model, selected_metric):
    filtered = long_df[(long_df['Model'] == selected_model) & (long_df['BaseMetric'] == selected_metric)]

    if selected_metric in ['Accuracy', 'Weighted F1']:
        fig = px.bar(
            filtered,
            x='Fold',
            y='Score',
            color='BaseMetric',
            title=f"{selected_metric} Across Folds ({selected_model})",
            labels={'Score': 'Score', 'Fold': 'Fold'},
            text_auto='.3f',
            color_discrete_sequence=['gray']
        )
        std_text = f"Average: {filtered['Score'].mean():.4f} | Standard Deviation: {filtered['Score'].std():.4f}"
    else:
        class0 = filtered[filtered['Class'] == 'Class 0']
        class1 = filtered[filtered['Class'] == 'Class 1']

        fig = go.Figure([
            go.Bar(name='Class 0', x=class0['Fold'], y=class0['Score'],
                   marker=dict(color='#3182bd')),
            go.Bar(name='Class 1', x=class1['Fold'], y=class1['Score'],
                   marker=dict(color='#08519c'))
        ])

        fig.update_layout(
            barmode='group',
            title=f"{selected_metric} Across Folds ({selected_model})",
            yaxis=dict(range=[0.8, 1.0]),
            xaxis_title='Fold',
            yaxis_title='Score'
        )

        # Determine metadata
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

        # Add below the legend (approx. bottom-right of plot area)
        fig.add_annotation(
            text=metadata_text,
            xref="paper", yref="paper",
            x=1.05,  # push it further right, just beyond the legend
            y=0.5,  # center vertically next to the plot (adjust if needed)
            showarrow=False,
            font=dict(size=10, color="#666"),
            align="left",
            textangle=0,
            valign="middle",
            xanchor="left",
            yanchor="middle"
        )

        std_text = (
            f"Average (Class 0): {class0['Score'].mean():.4f} | Standard Deviation (Class 0): {class0['Score'].std():.4f} | "
            f"Average (Class 1): {class1['Score'].mean():.4f} | Standard Deviation (Class 1): {class1['Score'].std():.4f}"
        )

    return fig, std_text


def update_table(selected_table):
    if selected_table == 'variance':
        table_df = variance_df
    else:
        table_df = display_prediction_df

    return dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in table_df.columns],
        data=table_df.to_dict('records'),
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f9f9f9'}
    )


def toggle_variance_collapse(n):
    return n % 2 == 1 if n else False


def create_variance_tile(base_path, overlay_path):
    return html.Div([
        html.Div([
            html.Img(src=f"/{base_path}", className="base"),
            html.Img(src=f"/{overlay_path}", className="overlay")
        ], className="tile-inner")
    ], className="tile")


def update_variance_samples(selected_family):
    df = variance_meta.query("family == @selected_family")  # No more 'All' logic
    df = df.reset_index(drop=True)

    tiles = []
    for i in range(0, len(df), 2):
        full_path = df.loc[i, 'full_path'].replace('app/', '')

        if '.txt.png' in full_path:
            idx = full_path.index('.txt.png') + len('.txt.png')
            base_path = full_path[:idx]
        else:
            continue

        overlay_path = base_path.replace('.txt.png', '.txt.png_var.png')

        tiles.append(create_variance_tile(base_path, overlay_path))

        if len(tiles) >= 16:
            break

    return tiles


# ------- CALLBACKS -------------
dash.get_app().callback(
    Output('bar-chart', 'figure'),
    Output('std-dev-display', 'children'),
    Input('model-selector', 'value'),
    Input('metric-dropdown', 'value')
)(update_chart)

dash.get_app().callback(
    Output('table-container', 'children'),
    Input('table-toggle', 'value')
)(update_table)

dash.get_app().callback(
    Output("collapse-container", "is_open"),
    Input("collapse-toggle", "n_clicks"),
    prevent_initial_call=True
)(lambda n: n % 2 == 1 if n else False)

dash.get_app().callback(
    Output("variance-collapse-container", "is_open"),
    Input("variance-collapse-toggle", "n_clicks"),
    prevent_initial_call=True
)(toggle_variance_collapse)

dash.get_app().callback(
    Output("variance-sample-grid", "children"),
    Input("variance-family-dd", "value")
)(update_variance_samples)

dash.get_app().callback(
    Output("variance-description", "children"),
    Input("variance-family-dd", "value")
)(lambda selected_family: f"16 randomly selected samples from {selected_family}, with highest variance segment outlined.")


#        dbc.Col(html.Div(right_plot_2, **{"data-aos": "fade-left"}), xs=12, md=6),
