import dash
from dash import html, dcc
import plotly.express as px
from content import home_text  # paragraph text content
import numpy as np
import plotly.graph_objects as go


dash.register_page(__name__, path='/')

# -------------   PLOTS -----------------

# Placeholder Bar chart
sample_barchart = dcc.Graph(
    figure=px.bar(x=["A", "B"], y=[1, 3]),  # replace this
    style={"width": "100%", "maxWidth": "800px"}
)

# Placeholder confusion matrix
conf_matrix = np.array([[195, 19],
                        [10, 215]])

labels = ['Benign', 'Malware']

conf_matrix_fig = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=labels,
    y=labels,
    colorscale='Blues',
    showscale=True,
    zmin=0,
    zmax=conf_matrix.max(),
    text=conf_matrix,
    texttemplate="%{text}",
    hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
))

conf_matrix_fig.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted",
    yaxis_title="True",
    xaxis=dict(side="bottom"),
    yaxis=dict(autorange="reversed")
)

conf_matrix_plot = dcc.Graph(figure=conf_matrix_fig, style={"maxWidth": "800px", "margin": "auto"})

# ------------- LAYOUT ----------------

layout = html.Div(
    [
        html.H5(" Project Motivation", className="heading"),
        html.Div(home_text.intro_paragraph, className="paragraph", **{"data-aos": "fade-up"}),
        html.Hr(),

        dcc.Graph(
            figure=px.bar(x=["A", "B"], y=[1, 3]),
            style={"width": "100%", "maxWidth": "800px"}
        ),


        html.Div(home_text.middle_paragraph, className="paragraph", **{"data-aos": "fade-up"}),

        dcc.Graph(
            figure=conf_matrix_fig,
            style={"maxWidth": "800px", "margin": "auto"}
        ),

        html.Div(home_text.final_paragraph, className="paragraph", **{"data-aos": "fade-up"})
    ],
    className="home-content"
)

