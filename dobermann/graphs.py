import typing as tp

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_candles_graph(
    ticker: str,
    candles: pd.DataFrame,
    extra_bottom_graph: tp.List[go.Figure] = None,
    extra_graphs: tp.List[go.Figure] = None
) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.15], vertical_spacing=0.05)

    if extra_graphs is None:
        extra_graphs = []

    candle_chart = go.Candlestick(
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
    )
    fig.add_trace(candle_chart, row=1, col=1)

    for graph in extra_graphs:
        fig.add_trace(graph, row=1, col=1)

    if extra_bottom_graph is None:
        extra_bottom_graph = [go.Bar(
            x=candles.index,
            y=candles.volume,
            marker_color='#7658e0',
            name='Volume',
        )]

    for graph in extra_bottom_graph:
        fig.add_trace(graph, row=2, col=1)

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


def draw_line(graph: go.Figure, x0: tp.Any, y0: tp.Any, x1: tp.Any, y1: tp.Any, opacity: float) -> None:
    graph.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1, line_color='#7658e0', opacity=opacity)


def draw_vline(graph: go.Figure, x: tp.Any, width: int = 1, opacity: float = 1) -> None:
    graph.add_vline(x=x, line_color='#7658e0', line_width=width, opacity=opacity)


def get_report_graph(*subgraphs: tp.List[go.Scatter], format_value: str = ',.0f'):
    eq_graph = go.Figure(subgraphs)
    # eq_graph.update_traces(mode='lines+markers')
    eq_graph.update_layout(
        {'plot_bgcolor': '#ffffff', 'paper_bgcolor': '#ffffff', 'legend_orientation': "h"},
        showlegend=True,
        yaxis_tickformat=format_value,
        yaxis_gridcolor='#f1f1f1',
        xaxis_showline=True,
        xaxis_mirror=True,
        yaxis_showline=True,
        yaxis_mirror=True,
        xaxis_linecolor='#f1f1f1',
        yaxis_linecolor='#f1f1f1',
        xaxis_color='grey',
        yaxis_color='grey',

        # Hover Line
        hovermode='x',
        xaxis_showspikes=True,
        xaxis_spikedash='dash',
        xaxis_spikemode='across',
        xaxis_spikesnap='cursor',
        xaxis_spikecolor='grey',
        xaxis_spikethickness=1,
    )
    return eq_graph


go.Layout
