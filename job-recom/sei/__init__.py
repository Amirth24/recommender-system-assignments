import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(
    name=__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=True,
    pages_folder="pages"
)


app.layout = [
    dbc.Navbar([
        dbc.NavbarBrand(html.Div([
            html.H3("Sei")

        ])),
    ], class_name="px-3"),
    dbc.Container([
        dash.page_container
    ])
]


def run_server(*args, **kwargs):
    app.run_server(*args, **kwargs)
