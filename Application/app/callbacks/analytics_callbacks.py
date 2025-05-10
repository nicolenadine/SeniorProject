"""
Callback functions for the analytics page.
"""

import dash
from dash.dependencies import Input, Output, State  # Added State import

from components.metrics_plots import create_overall_metric_chart, create_class_metric_chart
from components.data_tables import create_data_table
from components.variance_plots import create_variance_sample_tile


def register_analytics_callbacks(app, long_df, variance_df, display_prediction_df, variance_meta):
    """
    Register all callbacks for the analytics page.

    Args:
        app: Dash application instance
        long_df (DataFrame): Metrics data in long format
        variance_df (DataFrame): Variance data
        display_prediction_df (DataFrame): Prediction data
        variance_meta (DataFrame): Variance metadata

    Returns:
        None
    """
    print("Registering analytics callbacks...")  # Debug print

    @app.callback(
        Output('bar-chart', 'figure'),
        Output('std-dev-display', 'children'),
        Input('model-selector', 'value'),
        Input('metric-dropdown', 'value')
    )
    def update_chart(selected_model, selected_metric):
        """Update metrics chart based on selections."""
        print(f"Chart callback: model={selected_model}, metric={selected_metric}")  # Debug print

        # Filter data based on selections
        filtered = long_df[(long_df['Model'] == selected_model) &
                           (long_df['BaseMetric'] == selected_metric)]

        # Create appropriate chart type based on metric
        if selected_metric in ['Accuracy', 'Weighted F1']:
            return create_overall_metric_chart(filtered, selected_metric, selected_model)
        else:
            return create_class_metric_chart(filtered, selected_metric, selected_model)

    @app.callback(
        Output('table-container', 'children'),
        Input('table-toggle', 'value')
    )
    def update_table(selected_table):
        """Update data table based on selection."""
        print(f"Table callback: table={selected_table}")  # Debug print

        if selected_table == 'variance':
            return create_data_table(variance_df)
        else:
            return create_data_table(display_prediction_df)

    @app.callback(
        Output("collapse-container", "is_open"),
        Input("collapse-toggle", "n_clicks"),
        State("collapse-container", "is_open"),  # Added State parameter
        prevent_initial_call=True
    )
    def toggle_collapse(n_clicks, is_open):
        """Toggle table collapse section."""
        print(f"Toggle table callback: clicks={n_clicks}, is_open={is_open}")  # Debug print

        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("variance-collapse-container", "is_open"),
        Input("variance-collapse-toggle", "n_clicks"),
        State("variance-collapse-container", "is_open"),  # Added State parameter
        prevent_initial_call=True
    )
    def toggle_variance_collapse(n_clicks, is_open):
        """Toggle variance samples collapse section."""
        print(f"Toggle variance callback: clicks={n_clicks}, is_open={is_open}")  # Debug print

        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("variance-sample-grid", "children"),
        Input("variance-family-dd", "value")
    )
    def update_variance_samples(selected_family):
        """Update variance sample grid based on selected family."""
        print(f"Sample grid callback: family={selected_family}")  # Debug print

        # Filter data for selected family
        df = variance_meta.query("family == @selected_family").reset_index(drop=True)

        # Create sample tiles
        tiles = []
        for i in range(0, len(df), 2):
            if i >= len(df):
                break

            full_path = df.loc[i, 'full_path'].replace('app/', '')

            if '.txt.png' in full_path:
                idx = full_path.index('.txt.png') + len('.txt.png')
                base_path = full_path[:idx]
            else:
                continue

            overlay_path = base_path.replace('.txt.png', '.txt.png_var.png')

            # Create tile and add to list
            tiles.append(create_variance_sample_tile(base_path, overlay_path))

            # Limit to 16 samples
            if len(tiles) >= 16:
                break

        return tiles

    @app.callback(
        Output("variance-description", "children"),
        Input("variance-family-dd", "value")
    )
    def update_variance_description(selected_family):
        """Update variance description based on selected family."""
        print(f"Description callback: family={selected_family}")  # Debug print

        return f"16 randomly selected samples from {selected_family}, with highest variance segment outlined."

    print("All analytics callbacks registered successfully!")  # Debug print