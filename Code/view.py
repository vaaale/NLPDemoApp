from dash import dcc, html, dash_table
import plotly.graph_objs as go


def show_abstractive_qa(question, abstractive_api):
    if len(question) < 1:
        return html.Div([])

    answer, context = abstractive_api.answer(question)
    output = html.Div([
        html.H2("Question:"),
        html.P(question),
        html.Br(),
        html.H2("Answer:"),
        html.P(answer),
        html.H2("Context given to model:"),
        html.P(context)
    ])

    return output


def show_extractive_qa(question, api):
    if len(question) < 1:
        return html.Div([])

    response, context = api.exact_answer(question, top_k=3)
    return html.Div([
        html.Div([
            html.H2("Question:"),
            html.P(question),
            html.Br(),
            html.H2("Answer:"),
            html.P(response["answer"])
        ]),
        html.Div([
            html.H3("The context given to the model was:"),
            html.P(context)
        ])
    ])


def show_architecture():
    return html.Div([
        html.Img(id='bi_encoder_img', src="/static/Bi_Encoder.png"),
        html.Br(),
        html.Img(id='cross_encoder_img', src="/static/Cross_Encoder.png")
    ])


def show_answers(response, value):
    df_cross = response[['cross_encoder_scores', 'cross_encoder']].iloc[1:11]
    df_cross.columns = ['Distance', 'Text']
    df_bi = response[['bi_encoder_scores', 'bi_encoder']].iloc[1:11]
    df_bi['bi_encoder_scores'] = df_bi['bi_encoder_scores'].apply(lambda x: f"{x:.3f}")
    df_bi.columns = ['Distance', 'Text']
    df_bm25 = response[['bm25_scores', 'bm25']].iloc[1:11]
    df_bm25.columns = ['Distance', 'Text']
    output = html.Div([
        html.Div([
            html.H2("Question:"),
            html.P(value),
            html.Br(),
            html.H2("Answer:"),
            html.P(response["cross_encoder"][1])
        ]),
        html.Div([
            html.H2("Top 10 results from using keyword search"),
            dash_table.DataTable(
                df_bm25.to_dict("records"),
                [{"name": i, "id": i} for i in df_bm25.columns],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell={'textAlign': 'left'},
            ),
            html.H2("Top 10 results before filtering"),
            dash_table.DataTable(
                df_bi.to_dict("records"),
                [{"name": i, "id": i} for i in df_bi.columns],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell={'textAlign': 'left'},
            ),
            html.H2("Top 10 results after filtering"),
            dash_table.DataTable(
                df_cross.to_dict("records"),
                [{"name": i, "id": i} for i in df_cross.columns],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell={'textAlign': 'left'},
            ),
        ])
    ])
    return output


def show_embeddings(response, embeddings, api):
    emb = api.reduce(embeddings)
    response["bi_encoder_emb_X"] = emb[:, 0]
    response["bi_encoder_emb_Y"] = emb[:, 1]

    bi_encoder_emb_X = response["bi_encoder_emb_X"].iloc[1:]
    bi_encoder_emb_Y = response["bi_encoder_emb_Y"].iloc[1:]
    bi_encoder_scores = response["bi_encoder_scores"].iloc[1:]
    text = response["bi_encoder"].iloc[1:].apply(lambda x: x.replace('.', '.<br>'))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=bi_encoder_emb_X,
            y=bi_encoder_emb_Y,
            hovertext=text,
            marker=dict(
                color=bi_encoder_scores,
                size=bi_encoder_scores * 20,
            )
        )
    )
    q_bi_encoder_emb_X = response["bi_encoder_emb_X"].iloc[0:1]
    q_bi_encoder_emb_Y = response["bi_encoder_emb_Y"].iloc[0:1]
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=q_bi_encoder_emb_X,
            y=q_bi_encoder_emb_Y,
            marker=dict(
                color="green",
                size=20,
                opacity=0.5
            )
        )
    )
    fig.update_layout(
        width=1024,
        height=1024,
    )
    output = html.Div([
        html.H3("Embeddings of the 200 most relevant answers for the given question"),
        html.Br(),
        html.P("The green dot represents the question"),
        html.Br(),
        html.P("The yellow dots are the most relevant-, and the blue the least relevant answers"),
        dcc.Graph(figure=fig)
    ])
    return output

