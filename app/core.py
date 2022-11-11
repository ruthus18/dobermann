import datetime as dt
import enum
import typing as t
from decimal import Decimal

Asset = str


class Timeframe(enum.StrEnum):
    M5 = '5m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'

    @property
    def timedelta(self) -> dt.timedelta:
        return {
            self.M5: dt.timedelta(minutes=5),
            self.H1: dt.timedelta(hours=1),
            self.H4: dt.timedelta(hours=4),
            self.D1: dt.timedelta(days=1),
        }[self.value]


class Candle(t.TypedDict):
    open_time: dt.datetime
    open: float
    close: float
    low: float
    high: float
    volume: float


class TradeAction(enum.StrEnum):
    OPEN = 'open'
    CLOSE = 'close'


class TradeDirection(enum.StrEnum):
    BULL = 'bull'
    BEAR = 'bear'


class TradeEvent(t.TypedDict):
    # general for open/close event
    trade_id: int
    direction: TradeDirection
    size: Decimal  # in $

    # differ for open/close event
    time: dt.datetime
    action: TradeAction
    price: float
