import dash
from dash import html, dcc

from content import findings_text

dash.register_page(__name__, path="/findings", name="Findings")

layout = html.Div([
    html.Div([
        # Project Overview
        html.H4("Research Findings", className="heading"),

        html.Div(findings_text.overview_paragraph,
                 className="paragraph centered"),

        # Key Findings Section
        html.Hr(),
        html.H4("Key Findings", className="heading"),

        html.Div(findings_text.key_findings_intro,
                 className="paragraph left-align"),

        # Model Performance - Bulleted Section
        html.H6(findings_text.model_performance_title, className="subheading"),
        html.Div([
            html.Div(
                dcc.Markdown(findings_text.model_performance_bullet1,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.model_performance_bullet2,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.model_performance_bullet3,
                             dangerously_allow_html=True),
                className="bulleted-content"),
        ]),

        # Model Calibration - Bulleted Section
        html.H6(findings_text.calibration_title, className="subheading"),
        html.Div([
            html.Div(
                dcc.Markdown(findings_text.calibration_bullet1,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.calibration_bullet2,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.calibration_bullet3,
                             dangerously_allow_html=True),
                className="bulleted-content",),
        ]),

        # Segment Analysis - Bulleted Section
        html.H6(findings_text.segment_analysis_title, className="subheading"),
        html.Div([
            html.Div(
                dcc.Markdown(findings_text.segment_analysis_bullet1,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.segment_analysis_bullet2,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.segment_analysis_bullet3,
                             dangerously_allow_html=True),
                className="bulleted-content"),
        ]),

        # Computational Efficiency - Bulleted Section
        html.H6(findings_text.computational_efficiency_title,
                className="subheading"),

        html.Div([
            html.Div(
                dcc.Markdown(findings_text.computational_efficiency_bullet1,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.computational_efficiency_bullet2,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.computational_efficiency_bullet3,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.computational_efficiency_bullet4,
                             dangerously_allow_html=True),
                className="bulleted-content")
        ]),

        # Impact & Next Steps
        html.Hr(),
        html.H4("Impact & Future Directions", className="heading"),
        html.Hr(),

        # Impact Section - Bulleted

        html.H5(findings_text.impact_title, className="subheading"),

        html.Div([
            html.Div(
                dcc.Markdown(findings_text.impact_bullet1,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.impact_bullet2,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.impact_bullet3,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.impact_bullet4,
                             dangerously_allow_html=True),
                className="bulleted-content"),
        ]),

        # Next Steps Section - Bulleted
        html.H6(findings_text.next_steps_title, className="subheading"),
        html.Div([
            html.Div(
                dcc.Markdown(findings_text.next_steps_bullet1,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.next_steps_bullet2,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.next_steps_bullet3,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.next_steps_bullet4,
                             dangerously_allow_html=True),
                className="bulleted-content"),
            html.Div(
                dcc.Markdown(findings_text.next_steps_bullet5,
                             dangerously_allow_html=True),
                className="bulleted-content")
        ]),

        # Conclusion
        html.Hr(),
        html.H5("Conclusion", className="heading"),
        html.Div(findings_text.conclusion_paragraph,
                 className="paragraph centered"),

    ], className="findings-content")
], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "0 20px"})
