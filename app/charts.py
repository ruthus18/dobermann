import datetime as dt
import itertools
import math
import typing as t
import warnings
from decimal import Decimal

import altair as alt
import pandas as pd

from .core import Candle, TradeDirection, TradeEvent

warnings.filterwarnings('ignore', category=FutureWarning)

alt.data_transformers.disable_max_rows()


PASTEL_RED = '#ff6962'
PASTEL_GREEN = '#77dd76'


def _get_candles_zoom_x_selection() -> alt.Selection:
    return alt.selection_interval(bind='scales', encodings=['x'], zoom='wheel![event.shiftKey]')


def _get_candles_zoom_y_selection() -> alt.Selection:
    return alt.selection_interval(bind='scales', encodings=["y"], zoom='wheel![event.ctrlKey]')


def candles_chart(
    candles: list[Candle],
    *,
    width: int = 1200,
    height: int = 400,
    _zoom_x_selection: alt.Selection | None = None,
    _zoom_y_selection: alt.Selection | None = None,
    _hover_selection: alt.Selection | None = None,
) -> alt.LayerChart:
    df = pd.DataFrame.from_dict(candles).rename(columns={'open_time': 'time'})

    open_close_color = alt.condition(
        'datum.open <= datum.close', alt.value(PASTEL_GREEN), alt.value(PASTEL_RED)
    )
    base = alt.Chart(df).encode(alt.X('time:T', axis=alt.Axis(title='', grid=False)), color=open_close_color)

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

    if not _hover_selection:
        _hover_selection = _get_hover_selection()

    hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('open', format='.2f'),
            alt.Tooltip('low', format='.2f'),
            alt.Tooltip('high', format='.2f'),
            alt.Tooltip('close', format='.2f'),
        ])
        .add_selection(_hover_selection)
    )
    hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='time')
        .transform_filter(_hover_selection)
    )

    if not _zoom_x_selection:
        _zoom_x_selection = _get_candles_zoom_x_selection()

    if not _zoom_y_selection:
        _zoom_y_selection = _get_candles_zoom_y_selection()

    return (
        (candle_shadows + candle_bodies + hover_selectors + hover_rule)
        .properties(width=width, height=height)
        .add_selection(_zoom_x_selection)
        .add_selection(_zoom_y_selection)
    )


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
    _hover_selection: alt.Selection | None = None,
) -> alt.LayerChart:
    df = pd.DataFrame({
        'time': equities.keys(),
        'equity': equities.values(),
    })
    df.equity = df.equity.astype('float')

    equity_min_y_scale = (math.floor(df['equity'].min() / 100) - 0.5) * 100
    equity_max_y_scale = (math.ceil(df['equity'].max() / 100) + 0.5) * 100

    if not _hover_selection:
        _hover_selection = _get_hover_selection()

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
        .add_selection(_hover_selection)
    )
    equity_hover_points = (
        equity_line
        .mark_point(filled=True)
        .encode(opacity=alt.condition(_hover_selection, alt.value(1), alt.value(0)))
    )
    equity_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='time')
        .transform_filter(_hover_selection)
    )

    return alt.layer(
        equity_line, equity_start_line, equity_hover_selectors, equity_hover_rule, equity_hover_points
    ).properties(
        title='Equity', width=1200, height=250
    )


def leverage_chart(
    leverages: dict[dt.datetime, Decimal],
    _hover_selection: alt.Selection | None = None,
) -> alt.LayerChart:
    df = pd.DataFrame({
        'time': leverages.keys(),
        'leverage': leverages.values(),
    })
    df.leverage = df.leverage.astype('float')

    leverage_min_y_scale = 0
    leverage_max_y_scale = (math.ceil(df.leverage.max()) + 0.5)

    if not _hover_selection:
        _hover_selection = _get_hover_selection()

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
            opacity=alt.condition(_hover_selection, alt.value(1), alt.value(0.5))
        )
    )
    leverage_hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('leverage', format='.2f')
        ])
        .add_selection(_hover_selection)
    )
    leverage_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='time')
        .transform_filter(_hover_selection)
    )

    return alt.layer(
        leverage_hist, leverage_hover_selectors, leverage_hover_rule
    ).properties(
        title='Leverage Used', width=1200, height=150
    )


def drawdown_chart(
    drawdowns: dict[dt.datetime, Decimal],
    _hover_selection: alt.Selection | None = None,
) -> alt.LayerChart:
    df = pd.DataFrame({
        'time': drawdowns.keys(),
        'drawdown': drawdowns.values(),
    })
    df.drawdown = df.drawdown.astype('float')

    if not _hover_selection:
        _hover_selection = _get_hover_selection()

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
        .add_selection(_hover_selection)
    )
    drawdown_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='time')
        .transform_filter(_hover_selection)
    )
    drawdown_hover_points = (
        drawdown_line
        .mark_point(filled=True, color='red')
        .encode(opacity=alt.condition(_hover_selection, alt.value(1), alt.value(0)))
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

    return equity_chart_ & leverage_chart_ & drawdown_chart_


class CandlesView:

    def __init__(self, candles: list[Candle]):
        self._candles_data = candles
        self._overlay_data: dict[str, dict[dt.datetime, float]] = {}
        self._bottom_data: dict[str, dict[dt.datetime, float]] = {}

        self.width = 1200
        self.height = 400
        self.bottom_height_scale_factor = 3

    def add_overlay_data(self, signals: dict[dt.datetime, float], name: str) -> None:
        self._overlay_data[name] = signals

    def add_bottom_data(self, signals: dict[dt.datetime, float], name: str) -> None:
        self._bottom_data[name] = signals

    def _assert_time_series_consistency(self, timescales: t.Iterable[dt.datetime]) -> None:
        for T1, T2 in itertools.combinations(timescales, 2):
            assert T1 == T2, 'Time series are not the same!'

    def get_overlay_chart(self) -> alt.Chart | None:
        if not self._overlay_data:
            return None

        self._assert_time_series_consistency(sig.keys() for sig in self._overlay_data.values())  # type: ignore

        # We need only one time series so we take first
        # (it is not important which one, because we previously check that all time series are same)
        _first_name = list(self._overlay_data.keys())[0]
        chart_data = {
            'time': self._overlay_data[_first_name].keys(),
        }

        chart_data.update({
            name: signals.values()  # type: ignore
            for name, signals in self._overlay_data.items()
        })

        chart = (
            alt.Chart(pd.DataFrame(chart_data))
            .transform_fold(
                [name for name in self._overlay_data.keys()],
                as_=['signal', 'value']
            )
            .mark_line()
            .encode(x='time:T', y='value:Q', color='signal:N')
        )
        return chart

    def get_bottom_chart(self, _hover_selection: alt.Selection | None = None) -> alt.Chart | None:
        if not self._bottom_data:
            return None

        self._assert_time_series_consistency(sig.keys() for sig in self._bottom_data.values())  # type: ignore

        # We need only one time series so we take first
        # (it is not important which one, because we previously check that all time series are same)
        _first_name = list(self._bottom_data.keys())[0]
        chart_data = {
            'time': self._bottom_data[_first_name].keys(),
        }

        chart_data.update({
            name: signals.values()  # type: ignore
            for name, signals in self._bottom_data.items()
        })

        df = pd.DataFrame(chart_data)
        chart = (
            alt.Chart(df)
            .transform_fold(
                [name for name in self._bottom_data.keys()],
                as_=['signal', 'value']
            )
            .mark_line()
            .encode(
                x=alt.X('time:T', axis=alt.Axis(title='', grid=False)),
                y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
                color='signal:N'
            )
            .properties(width=self.width, height=self.height // self.bottom_height_scale_factor)
        )

        if not _hover_selection:
            _hover_selection = _get_hover_selection()

        hover_selectors = (
            alt.Chart(df)
            .mark_rule(size=10)
            .encode(x='time', opacity=alt.value(0), tooltip=[
                alt.Tooltip('time', format='%d %b %Y, %H:%M'),
                *[
                    alt.Tooltip(name, format='.2f')
                    for name in self._bottom_data.keys()
                ]
            ])
            .encode(x='time', opacity=alt.value(0))
            .add_selection(_hover_selection)
        )
        hover_rule = (
            alt.Chart(df)
            .mark_rule(opacity=0.25, size=0.25)
            .encode(x='time')
            .transform_filter(_hover_selection)
        )

        return chart + hover_selectors + hover_rule

    @property
    def chart(self) -> alt.Chart | alt.VConcatChart:
        zoom_x = _get_candles_zoom_x_selection()
        zoom_y = _get_candles_zoom_y_selection()
        hover_selection = _get_hover_selection()

        chart = candles_chart(self._candles_data, _zoom_x_selection=zoom_x, _hover_selection=hover_selection)

        # Check data variable to avoid building charts twice (this is redundant)

        if self._overlay_data:
            chart += self.get_overlay_chart()

        if self._bottom_data:
            chart &= (
                self.get_bottom_chart(_hover_selection=hover_selection)  # type: ignore
                .add_selection(zoom_x)
                .add_selection(zoom_y)
            )

        return chart.resolve_scale(color='independent')
