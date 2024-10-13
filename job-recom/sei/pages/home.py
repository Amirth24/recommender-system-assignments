from dash import html, register_page
import dash_bootstrap_components as dbc
from sei.user_form import form

register_page(__name__, path="")

layout = html.Div([
    html.H1("Find the job that is meant for You!"),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                form
            ], class_name="w-50"),
            dbc.Col([

                html.Div(id="output-recommendation")

            ])

        ])
    ], class_name="mt-5"),
])
