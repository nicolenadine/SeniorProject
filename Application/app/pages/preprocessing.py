import dash
from dash import html, dcc
import plotly.express as px

from content import preprocessing_text
from content.data_sources import opcode_embeddings_tsne_img, sample_images_img


dash.register_page(__name__, path="/preprocessing", name="Preprocessing")

image_style = {"width": "100%", "paddingLeft": "20px", "paddingTop": "25px"}

# T-sne image
tsne_image = html.Img(
    src=opcode_embeddings_tsne_img,
    style={**image_style, 'height': '700px'},
    alt="t-SNE visualization of opcode embeddings")

# Grid of sample malware visualizations
sample_image = html.Img(
    src=sample_images_img,
    style={**image_style, 'height': '375px'},
    alt="Sample Hilbert curve visualizations of malware files")

layout = html.Div([

    # Section 1: Opcode sequence extraction paragraph
    html.Hr(),
    html.H5("Opcode Sequence Extraction", className="heading"),
    html.Hr(),

    html.Div(preprocessing_text.opcode_extraction,
             className="paragraph left-align"),

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
