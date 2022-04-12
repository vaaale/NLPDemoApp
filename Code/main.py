import dash
import dash_bootstrap_components as dbc
import flask
from dash import Input, Output, dcc, html

from api import ExtractiveAPI, AbstractiveAPI
from view import show_answers, show_embeddings, show_architecture, show_extractive_qa, show_abstractive_qa

global response
global embeddings
global extractive_api
global abstractive_api

response = None
embeddings = None

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
        dcc.Textarea(
            id='query-text',
            value='what is the capital of Norway?',
            placeholder='Ask me something',
            style={'width': '100%', 'height': 300},
        ),
        html.Button('Submit question', id='submit-query'),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div([
    dcc.Tabs(id="tabs-container", value='answer-tab', children=[
        dcc.Tab(label='Answers', value='answer-tab'),
        dcc.Tab(label='Embeddings', value='embedding-tab'),
        dcc.Tab(label='Extractive QA', value='extractive-qa-tab'),
        dcc.Tab(label='Abstractive QA', value='abstractive-qa-tab'),
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
def update_output(question, n_clicks, tab):
    global response
    global embeddings

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "submit-query.n_clicks" in changed_id:
        response, embeddings = extractive_api.score(query=question, top_k=200)

    if response is None:
        return html.Div([])

    if tab == 'answer-tab':
        return show_answers(response, question)
    elif tab == 'embedding-tab':
        return show_embeddings(response, embeddings, extractive_api)
    elif tab == 'extractive-qa-tab':
        return show_extractive_qa(question, extractive_api)
    elif tab == 'abstractive-qa-tab':
        return show_abstractive_qa(question, abstractive_api)
    else:
        return show_architecture()


@server.route("/static/<image_path>", methods=['GET', 'POST'])
def serve_image(image_path):
    output = flask.send_from_directory("../Assets/", image_path)
    print("Returning image")
    return output


if __name__ == "__main__":
    print("Creating API")
    extractive_api = ExtractiveAPI()
    abstractive_api = AbstractiveAPI()

    print("Starting server")
    app.run_server(port=8050, debug=True, use_reloader=False)
