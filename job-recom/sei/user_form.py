from recommender import get_recommendation
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, callback, dcc


def get_form(user=None):
    return [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("Experience", html_for="user-exp"), width=2),
                dbc.Col(
                    dbc.Input(
                        type="number",
                        id="user-exp",
                        placeholder="Enter your Experience in years"
                    ),
                    width=10,
                ),
            ],
            className="mb-3"
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("About Yourself", html_for="about-user"), width=2),
                dbc.Col(
                    dbc.Input(
                        type="textarea",
                        id="about-user",
                        placeholder="Tell Us about yourself"
                    ),
                    width=10,
                ),
            ],
            className="mb-3"
        ),
        dbc.Button("Submit", id="submit-button",
                   color="primary", n_clicks=0),
    ]


form = dbc.Container(
    [
        dbc.Row([
            html.H3("Enter Your Details"),
        ]),
        dbc.Form(
            get_form(),
            id="user-form"
        ),
    ],
    fluid=True,
)


@callback(
    Output("output-recommendation", "children"),
    [Input("submit-button", "n_clicks")],
    [State("user-exp", "value"),
     State("about-user", "value")]
)
def update_output(n_clicks, exp, about):
    if n_clicks > 0:
        # This is a placeholder, you can replace it with actual job recommendation logic

        if exp is None:
            exp = 0

        recommendations = get_recommendation(exp, about)

        return dbc.ListGroup([
            dbc.ListGroupItem(dbc.Card(children=[
                dbc.CardHeader(
                    [html.H3(job[1].job_title),
                     html.B([job[1].min_exp_yrs, " YOE is Required"])]
                ),
                dbc.Row(
                    html.Iframe(srcDoc=job[1].html_job_description)
                ),
                dbc.CardFooter([
                    dbc.Row([
                        html.Span([job[1].state, ", ", job[1].country]),
                    ]),
                    dbc.Row([

                        html.P(job[1].postdate_in_indexname_format)
                    ]),
                ]
                ),

            ]))
            for job in recommendations.iterrows()
        ])
