import datetime as dt
from dataclasses import dataclass

import pytz

from .binance_client import Timeframe
from .config import settings


TZ = pytz.timezone(settings.TZ_NAME)


@dataclass(slots=True)
class Candle:
    timeframe: Timeframe
    open_time: dt.datetime
    open: float
    close: float
    low: float
    high: float
    volume: float

    @classmethod
    def from_dict(cls, d: dict) -> 'Candle':
        return cls(**{
            **d,
            'open_time': d['open_time'].astimezone(TZ),
            'open': float(d['open']),
            'close': float(d['close']),
            'high': float(d['high']),
            'low': float(d['low']),
            'volume': float(d['volume']),
        })
