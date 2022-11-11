import datetime as dt
import typing as t

from .core import Candle
from .indicators import _V, Indicator


class FeedItem(t.TypedDict, t.Generic[_V]):
    time: dt.datetime
    value: _V


def feed(candles: list[Candle], indicator: Indicator[_V]) -> list[FeedItem[_V | None]]:
    out = [
        FeedItem(
            time=candle['open_time'],
            value=indicator.calculate(candle['close']),
        )
        for candle in candles
    ]
    indicator.reset()
    return out
