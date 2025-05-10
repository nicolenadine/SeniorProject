import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
from content import objectives_text
from content.data_sources import benign_file_count_img, top_malware_families_img

dash.register_page(__name__, path="/project_objectives", name="Project Objectives")

image_style = {"width": "100%", "paddingLeft": "20px", "paddingTop": "25px"}

# Benign sample count bar chart
benign_bar = html.Img(
                src= benign_file_count_img,
                style={**image_style, 'height': '350px'},
                alt="Benign file count by source")


# Malware sample count bar chart
malware_bar = html.Img(
                src= top_malware_families_img,
                style={**image_style, 'height': '450px'},
                alt="Benign file count by source")

# --- Layout ---
layout = html.Div([

    html.H5("Background Motivation & Project Objectives", className="heading"),

    # Intro Paragraphs (centered)
    html.Div(objectives_text.intro_cnn_paragraph, className="paragraph centered"),
    html.Div(objectives_text.intro_cnn_paragraph2, className="paragraph centered"),
    html.Div(objectives_text.segmentation_goal_paragraph, className="paragraph centered"),

    html.Hr(),
    html.H5("Data Sources", className="heading"),
    html.Hr(),

    # Malware: Left paragraph + Right chart
    dbc.Row([
        dbc.Col(html.Div(malware_bar), xs=16, md=8),
        dbc.Col(html.Div(objectives_text.malware_paragraph,
                         className="paragraph centered left-align"), xs=8,md=4)
    ], className="mb-4", style={"padding-bottom": "20px"}),

    html.Hr(),

    # Benign: Left paragraph + Right chart
    dbc.Row([
        dbc.Col(html.Div(objectives_text.benign_paragraph,
                         className="paragraph centered left-align"), xs=8,
                md=4),
        dbc.Col(html.Div(benign_bar), xs=16, md=8),
    ], className="mb-4", style={"padding-top": "20px"}),

    html.Hr(),

    html.Div([
        html.H5("References", className="references-title"),
        html.Ul([
            html.Li(citation, className="citation") for citation in objectives_text.citations
        ])
    ], className="references-section")

], className="objectives-content")
