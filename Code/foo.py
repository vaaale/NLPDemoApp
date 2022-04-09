# Based on the example from https://www.activestate.com/blog/dash-vs-bokeh/
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as obj
import uvicorn as uvicorn
from dash.dependencies import Input, Output
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

app = dash.Dash(__name__, requests_pathname_prefix="/dash/")

years = list(range(1940, 2021, 1))
temp_high = [x / 20 for x in years]
temp_low = [x - 20 for x in temp_high]
df = pd.DataFrame({"Year": years, "TempHigh": temp_high, "TempLow": temp_low})

slider = dcc.RangeSlider(
    id="slider",
    value=[df["Year"].min(), df["Year"].max()],
    min=df["Year"].min(),
    max=df["Year"].max(),
    step=5,
    marks={
        1940: "1940",
        1945: "1945",
        1950: "1950",
        1955: "1955",
        1960: "1960",
        1965: "1965",
        1970: "1970",
        1975: "1975",
        1980: "1980",
        1985: "1985",
        1990: "1990",
        1995: "1995",
        2000: "2000",
        2005: "2005",
        2010: "2010",
        2015: "2015",
        2020: "2020",
    },
)

app.layout = html.Div(
    children=[
        html.H1(children="Data Visualization with Dash"),
        html.Div(children="High/Low Temperatures Over Time"),
        dcc.Graph(id="temp-plot"),
        slider,
    ]
)


@app.callback(Output("temp-plot", "figure"), [Input("slider", "value")])
def add_graph(slider):
    print(type(slider))
    trace_high = obj.Scatter(x=df["Year"], y=df["TempHigh"], mode="markers", name="High Temperatures")
    trace_low = obj.Scatter(x=df["Year"], y=df["TempLow"], mode="markers", name="Low Temperatures")
    layout = obj.Layout(xaxis=dict(range=[slider[0], slider[1]]), yaxis={"title": "Temperature"})
    figure = obj.Figure(data=[trace_high, trace_low], layout=layout)
    return figure


if __name__ == "__main__":
    server = FastAPI()
    server.mount("/dash", WSGIMiddleware(app.server))
    uvicorn.run(server)