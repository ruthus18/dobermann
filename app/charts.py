import datetime as dt
import math
import warnings
from decimal import Decimal

import altair as alt
import pandas as pd

from .core import Candle, TradeDirection, TradeEvent

warnings.filterwarnings('ignore', category=FutureWarning)


PASTEL_RED = '#ff6962'
PASTEL_GREEN = '#77dd76'


def candles_chart(candles: list[Candle], *, width: int = 1200, height: int = 300) -> alt.LayerChart:
    df = pd.DataFrame.from_dict(candles)

    open_close_color = alt.condition(
        'datum.open <= datum.close', alt.value(PASTEL_GREEN), alt.value(PASTEL_RED)
    )
    base = alt.Chart(df).encode(alt.X('open_time:T'), color=open_close_color)

    candle_shadows = base.mark_rule().encode(
        alt.Y(
            'low:Q',
            title='Price',
            scale=alt.Scale(zero=False),
        ),
        alt.Y2('high:Q')
    )
    candle_bodies = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )
    return (candle_shadows + candle_bodies).properties(width=width, height=height).interactive()


def line_indicator_chart(
    signals: dict[dt.datetime, float], *, color: str = 'black', size: float = 1, opacity: float = 0.5
) -> alt.LayerChart:
    """Draw numeric indicator as single line
    """
    df = pd.DataFrame({
        'time': signals.keys(),
        'value': signals.values(),
    })
    return alt.Chart(df).mark_line(color=color, opacity=opacity, size=size).encode(x='time', y='value')


def bin_indicator_chart(signals: dict[dt.datetime, bool]) -> alt.LayerChart:
    """Draw binary indicator (True/False values) as vertical green/red lines
    """
    df = pd.DataFrame({
        'time': signals.keys(),
        'value': signals.values(),
    })
    return (
        alt.Chart(
            df[df.value.notnull()]
        )
        .mark_rule(size=3, opacity=0.3)
        .encode(
            x='time',
            color=alt.condition(alt.datum.value == True, alt.value('green'), alt.value('red'))  # noqa
        )
    )


def trades_chart(trades: list[TradeEvent], *, label_size: int = 16) -> alt.LayerChart:
    STONKS_UP = '▲'
    STONKS_DOWN = '▼'

    df = pd.DataFrame.from_dict(trades)
    text_cond = alt.condition(
        alt.FieldEqualPredicate(field='direction', equal=TradeDirection.BULL),
        alt.value(STONKS_UP),
        alt.value(STONKS_DOWN),
    )

    return alt.Chart(df).mark_text().encode(x='time', y='price', text=text_cond, size=alt.value(label_size))


def _get_hover_selection() -> alt.Selection:
    return alt.selection(type='single', nearest=True, on='mouseover', fields=['time'], empty='none')


def equity_chart(
    equities: dict[dt.datetime, Decimal],
    hover_selection: alt.Selection | None = None,
) -> alt.LayerChart:
    df = pd.DataFrame({
        'time': equities.keys(),
        'equity': equities.values(),
    })
    df.equity = df.equity.astype('float')

    equity_min_y_scale = (math.floor(df['equity'].min() / 100) - 0.5) * 100
    equity_max_y_scale = (math.ceil(df['equity'].max() / 100) + 0.5) * 100

    if not hover_selection:
        hover_selection = _get_hover_selection()

    equity_line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(
                'time',
                axis=alt.Axis(grid=False, title='')
            ),
            y=alt.Y(
                'equity',
                axis=alt.Axis(title=''),
                scale=alt.Scale(domain=[equity_min_y_scale, equity_max_y_scale])
            ),
        )
        .transform_filter(filter={'field': 'equity', 'valid': True})
    )
    equity_start_line = (
        alt.Chart(pd.DataFrame({'y': [1000]}))
        .mark_rule(size=0.2)
        .encode(y='y', color=alt.value('red'))
    )
    equity_hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('equity', format='.2f')
        ])
        .add_selection(hover_selection)
    )
    equity_hover_points = (
        equity_line
        .mark_point(filled=True)
        .encode(opacity=alt.condition(hover_selection, alt.value(1), alt.value(0)))
    )
    equity_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='time')
        .transform_filter(hover_selection)
    )

    return alt.layer(
        equity_line, equity_start_line, equity_hover_selectors, equity_hover_rule, equity_hover_points
    ).properties(
        title='Equity', width=1200, height=250
    )


def leverage_chart(
    leverages: dict[dt.datetime, Decimal],
    hover_selection: alt.Selection | None = None,
) -> alt.LayerChart:
    df = pd.DataFrame({
        'time': leverages.keys(),
        'leverage': leverages.values(),
    })
    df.leverage = df.leverage.astype('float')

    leverage_min_y_scale = 0
    leverage_max_y_scale = (math.ceil(df.leverage.max()) + 0.5)

    if not hover_selection:
        hover_selection = _get_hover_selection()

    leverage_hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                'time',
                axis=alt.Axis(grid=False, title='')
            ),
            y=alt.Y(
                'leverage',
                axis=alt.Axis(title=''),
                scale=alt.Scale(domain=[leverage_min_y_scale, leverage_max_y_scale])
            ),
            color=alt.value('darkgreen'),
            opacity=alt.condition(hover_selection, alt.value(1), alt.value(0.5))
        )
    )
    leverage_hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('leverage', format='.2f')
        ])
        .add_selection(hover_selection)
    )
    leverage_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='time')
        .transform_filter(hover_selection)
    )

    return alt.layer(
        leverage_hist, leverage_hover_selectors, leverage_hover_rule
    ).properties(
        title='Leverage Used', width=1200, height=150
    )


def drawdown_chart(
    drawdowns: dict[dt.datetime, Decimal],
    hover_selection: alt.Selection | None = None,
) -> alt.LayerChart:
    df = pd.DataFrame({
        'time': drawdowns.keys(),
        'drawdown': drawdowns.values(),
    })
    df.drawdown = df.drawdown.astype('float')

    if not hover_selection:
        hover_selection = _get_hover_selection()

    drawdown_line = (
        alt.Chart(df)
        .mark_line(color='red')
        .encode(
            x=alt.X('time', axis=alt.Axis(grid=False, title='')),
            y=alt.Y('drawdown', axis=alt.Axis(title='')),
        )
        .transform_filter(filter={'field': 'drawdown', 'valid': True})
    )
    drawdown_hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('drawdown', format='.6f')
        ])
        .add_selection(hover_selection)
    )
    drawdown_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='time')
        .transform_filter(hover_selection)
    )
    drawdown_hover_points = (
        drawdown_line
        .mark_point(filled=True, color='red')
        .encode(opacity=alt.condition(hover_selection, alt.value(1), alt.value(0)))
    )

    return alt.layer(
        drawdown_line, drawdown_hover_selectors, drawdown_hover_rule, drawdown_hover_points
    ).properties(
        title='Drawdown', width=1200, height=100
    )


def equity_leverage_drawdown_chart(
    equities: dict[dt.datetime, Decimal],
    leverages: dict[dt.datetime, Decimal],
    drawdowns: dict[dt.datetime, Decimal],
) -> alt.VConcatChart:
    hover_selection = _get_hover_selection()

    equity_chart_ = equity_chart(equities, hover_selection)
    leverage_chart_ = leverage_chart(leverages, hover_selection)
    drawdown_chart_ = drawdown_chart(drawdowns, hover_selection)

    return alt.vconcat(equity_chart_, leverage_chart_, drawdown_chart_)
