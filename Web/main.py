from contextlib import redirect_stderr

import dash
import dash_bootstrap_components as dbc
import flask
from dash import Input, Output, dcc, html, dash_table

from Web.view import show_answers, show_embeddings, show_architecture
from api import score
import plotly.express as px
import plotly.graph_objs as go

global response
response = None

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "32rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",

}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "34rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("Q&A", className="display-4"),
        html.Hr(),
        html.P("I will be using BERT and the power of Cross-encoding to answer your questions.", className="lead"),
        html.P("Go ahead and ask me anything!", className="lead"),
        # dcc.Input(value='', type='text', size="40"),
        dcc.Textarea(
            id='query-text',
            value='what is the capital of Norway',
            placeholder='Ask me something',
            style={'width': '100%', 'height': 300},
        ),
        html.Button('Submit question', id='submit-query'),
    ],
    style=SIDEBAR_STYLE,
)

# content = html.Div(id="page-content", style=CONTENT_STYLE)
content = html.Div([
    # html.H1('Demo stuff'),
    dcc.Tabs(id="tabs-container", value='answer-tab', children=[
        dcc.Tab(label='Answers', value='answer-tab'),
        dcc.Tab(label='Embeddings', value='embedding-tab'),
        dcc.Tab(label='Architecture', value='architecture-tab'),
    ]),
    html.Div(id="page-content")
], style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output("page-content", "children"),
    Input('query-text', 'value'),
    Input(component_id='submit-query', component_property='n_clicks'),
    Input('tabs-container', 'value')
)
def update_output(value, n_clicks, tab):
    global response
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "submit-query.n_clicks" in changed_id:
        response = score(query=value, top_k=200)

    if response is None:
        return html.Div([])

    if tab == 'answer-tab':
        return show_answers(response, value)
    elif tab == 'embedding-tab':
        return show_embeddings(response)
    else:
        return show_architecture()


@server.route("/static/<image_path>", methods=['GET', 'POST'])
def serve_image(image_path):
    output = flask.send_from_directory("../Assets/", image_path)
    print("Returning image")
    return output


if __name__ == "__main__":
    app.run_server(port=8050, debug=True)
