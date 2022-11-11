import warnings

import altair as alt
import pandas as pd

from .core import Candle, TradeDirection, TradeEvent

warnings.filterwarnings('ignore', category=FutureWarning)


class Color:
    green = '#06982d'
    red = '#ae1325'
    pastel_red = '#ff6962'
    pastel_green = '#77dd76'
    pastel_light_red = '#ffb6b3'
    pastel_light_green = '#bde7bd'


def candles_chart(candles: list[Candle], *, width: int = 1200, height: int = 300) -> alt.LayerChart:
    df = pd.DataFrame.from_dict(candles)

    open_close_color = alt.condition(
        'datum.open <= datum.close', alt.value(Color.pastel_green), alt.value(Color.pastel_red)
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
