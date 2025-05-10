"""
Components for displaying data tables.
"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc  # Correct import for dbc


def create_data_table(data_df):
    """
    Create a data table component.

    Args:
        data_df (DataFrame): Data to display in the table

    Returns:
        dash_table.DataTable: Table component
    """
    return dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in data_df.columns],
        data=data_df.to_dict('records'),
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f9f9f9'}
    )


def create_collapsible_tables_section(variance_df, prediction_df):
    """
    Create a collapsible section containing data tables.

    Args:
        variance_df (DataFrame): Variance data
        prediction_df (DataFrame): Prediction data without file_path column

    Returns:
        html.Div: Collapsible tables section
    """
    return html.Div([
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
    ], className="mb-4")