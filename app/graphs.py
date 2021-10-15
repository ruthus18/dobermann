from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_candles_graph(ticker: str, candles: pd.DataFrame, extra_graph: Optional[go.Figure] = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.15], vertical_spacing=0.05)

    fig.add_trace(
        go.Candlestick(
            x=candles.open_time,
            open=candles.open,
            high=candles.high,
            low=candles.low,
            close=candles.close,
            name=ticker,
            increasing_line_color='#2d9462',
            increasing_fillcolor='#2d9462',
            decreasing_line_color='#f62f47',
            decreasing_fillcolor='#f62f47',
            line={'width': 1},
            yhoverformat='%{y:.10f}'
        ),
        row=1, col=1,
    )
    if extra_graph is not None:
        second_graph = extra_graph
    else:
        second_graph = go.Bar(
            x=candles.open_time,
            y=candles.volume,
            marker_color='#7658e0',
            name='Volume',
        )
    fig.add_trace(second_graph, row=2, col=1)
    fig.update_layout(
        {'plot_bgcolor': '#ffffff', 'paper_bgcolor': '#ffffff', 'legend_orientation': "h"},
        legend=dict(y=1, x=0),
        height=700,
        hovermode='x unified',
        margin=dict(b=20, t=0, l=0, r=40),
        bargap=0.1,
    )

    axes_config = {
        'zeroline': False,
        'showgrid': False,
        'showline': False,
        'showspikes': True,
        'spikemode': 'across',
        'spikedash': 'solid',
        'spikesnap': 'cursor',
        'spikecolor': '#aaaaaa',
        'spikethickness': 1,
    }
    fig.update_yaxes(**axes_config)
    fig.update_xaxes(rangeslider_visible=False, **axes_config)

    fig.update_traces(xaxis='x', hoverinfo='x+y')
    return fig


def update_graph_hover(graph: go.Figure, show_hover: bool) -> None:
    graph.update_layout(hoverdistance=1 if show_hover else 0)


def draw_line(graph: go.Figure, x0: Any, y0: Any, x1: Any, y1: Any, opacity: float) -> None:
    graph.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1, line_color='#7658e0', opacity=opacity)


def draw_vline(graph: go.Figure, x: Any, width: int = 1, opacity: float = 1) -> None:
    graph.add_vline(x=x, line_color='#7658e0', line_width=width, opacity=opacity)
