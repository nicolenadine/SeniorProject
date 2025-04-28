import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[
    dbc.themes.SIMPLEX], suppress_callback_exceptions=True)

server = app.server

# App title
title = html.Div([
    html.Span("Malware Classification", style={"fontWeight": "bold"}),
    html.Span(": Exploring CNN Image Segmentation")
], className="navbar-title")

# Nav links
nav_links = dbc.Nav([
    dbc.NavItem(dcc.Link("Home", href="/", className="nav-link")),
    dbc.NavItem(dcc.Link("Objectives", href="/project_objectives", className="nav-link")),
    dbc.NavItem(dcc.Link("Preprocessing", href="/preprocessing",
                         className="nav-link")),
    dbc.NavItem(dcc.Link("Analytics", href="/analytics", className="nav-link")),
    dbc.NavItem(dcc.Link("Findings", href="/findings", className="nav-link")),
], className="ms-auto", navbar=True)

# Responsive Navbar layout
navbar = dbc.Navbar(
    dbc.Container([
        title,
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(nav_links, id="navbar-collapse", is_open=False, navbar=True),
    ], fluid=True),
    color="light",
    light=True,
    sticky="top"
)


# Toggle callback
@app.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open")
)
def toggle_navbar(n, is_open):
    if n:
        return not is_open
    return is_open


# Layout with page container
app.layout = html.Div([
    navbar,
    dash.page_container
])

if __name__ == "__main__":
    app.run(debug=True)
