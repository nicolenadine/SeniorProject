import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
from content import preprocessing_text

dash.register_page(__name__, path="/preprocessing", name="Preprocessing")

# Placeholder bar chart
bar_chart = dcc.Graph(
    figure=px.bar(x=["Step A", "Step B"], y=[10, 15]),
    style={"width": "100%"}
)

# Placeholder scatter plot
scatter_plot = dcc.Graph(
    figure=px.scatter(x=[1, 2, 3], y=[4, 1, 6]),
    style={"width": "100%"}
)

# Placeholder image
tsne_image = html.Img(
                src="https://github.com/nicolenadine/SeniorProject/blob/main/plots/opcode_embeddings_tsne.png?raw=true",
                style={'width': '100%', 'height': '700px', "padding-left":
                       "20px", "padding-top": "25px"},
                alt="Benign file count by source")
sample_image = html.Img(
                src="https://github.com/nicolenadine/SeniorProject/blob/main/plots/sample_images.png?raw=true",
                style={'width': '100%', 'height': '375px', "padding-left":
                       "20px", "padding-top": "25px"},
                alt="Benign file count by source")

layout = html.Div([

    # Section 1: Opcode sequence extraction paragraph
    html.Hr(),
    html.H5("Opcode Sequence Extraction", className="heading"),
    html.Hr(),

    html.Div(preprocessing_text.opcode_extraction, className="paragraph left-align"),

    html.Hr(),

    # Section 2: Word2Vec  Paragraph then tsne plot image
    html.Div(preprocessing_text.word2vec, className="paragraph left-align"),

    html.Div(tsne_image, style={"marginBottom": "1rem"}),

    # Section 3: Hilbert Mapping
    html.Hr(),
    html.H5("Hilbert Curve Mapping", className="heading"),
    html.Hr(),

    html.Div(preprocessing_text.hilbert_mapping, className="paragraph "
                                                           "left-align"),
    html.Div(sample_image, style={"marginBottom": "1rem"}),

    html.Hr(),

    # Section 4: Data Sampling
    html.Hr(),
    html.H5("Data Sampling", className="heading"),
    html.Hr(),

    html.Div(preprocessing_text.data_sampling, className="paragraph "
                                                           "left-align")
],
    style={"maxWidth": "1100px", "margin": "auto"}
)

