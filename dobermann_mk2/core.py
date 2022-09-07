import enum
import datetime as dt
from functools import cached_property
import typing as tp
from decimal import Decimal


class StrEnum(str, enum.Enum):
    def __str__(self):
        return self.value


Asset = str


class Timeframe(StrEnum):
    M5 = '5m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'

    @cached_property
    def timedelta(self) -> dt.timedelta:
        return {
            self.M5: dt.timedelta(minutes=5),
            self.H1: dt.timedelta(hours=1),
            self.H4: dt.timedelta(hours=4),
            self.D1: dt.timedelta(days=1),
        }[self.value]


class Candle(tp.TypedDict):
    open_time: dt.datetime
    open: float
    close: float
    low: float
    high: float
    volume: float


class TradeAction(StrEnum):
    OPEN = 'open'
    CLOSE = 'close'


class TradeDirection(StrEnum):
    BULL = 'bull'
    BEAR = 'bear'


class TradeEvent(tp.TypedDict):
    # general for open/close event
    trade_id: int
    action: TradeAction
    direction: TradeDirection
    size: Decimal  # in $

    # differ for open/close event
    time: dt.datetime
    price: Decimal
