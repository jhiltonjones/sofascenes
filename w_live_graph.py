import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

CSV = "/home/jack/sofascenes/contact_angle5.csv"

app = Dash(__name__)
app.layout = html.Div([
    html.H3("Contact angle around circumference (radius = time)"),
    dcc.Graph(id="polar"),
    dcc.Interval(id="timer", interval=200, n_intervals=0),  # ms
])

@app.callback(Output("polar", "figure"), Input("timer", "n_intervals"))
def update(_):
    try:
        df = pd.read_csv(CSV)
        if "has_contact" in df.columns:
            df = df[df["has_contact"].astype(int) == 1]
        df = df[np.isfinite(df["theta_wrapped"].astype(float))]
        fig = px.scatter_polar(df, r="t", theta="theta_wrapped")
        fig.update_layout(showlegend=False)
        return fig
    except Exception:
        return px.scatter_polar(pd.DataFrame({"t": [], "theta_wrapped": []}), r="t", theta="theta_wrapped")

if __name__ == "__main__":
    app.run_server(debug=False, host="127.0.0.1", port=8050)
