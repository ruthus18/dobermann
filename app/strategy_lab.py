# flake8: noqa
import datetime as dt

import numpy as np

from .core import Candle, Timeframe, TradeEvent
from .db import get_candles
from .indicators import Indicator

# Constraints:
#    * operating only one asset and one timeframe
#    * start_at, end_at = const


# Core part of backtester
def generate_signals(candles: list[Candle], indicator: Indicator) -> np.ndarray:
    indicator.reset()

    signals = []
    for candle in candles:
        signal = indicator.calculate(candle.close)
        signals.append(signal)

    return np.ndarray(signals)


def generate_long_short_trade_events(candles: list[Candle]) -> list[TradeEvent]:
    ...


def account_report(trades: list[TradeEvent]) -> ...:
    INITIAL_EQUITY = ...
    BROKER_COMISSION = ...
    # other params like round-trip cost etc...
    ...


async def showcase() -> None:
    asset = 'BTCUSD'
    timeframe = Timeframe.M5
    start_at = dt.datetime(2022, 1, 1)
    end_at = dt.datetime(2022, 7, 1)

    candles = await get_candles(asset, timeframe, start_at, end_at)
    ema_9 = Indicator.EMA(size=9)

    sig_ema_9 = generate_signals(candles, ema_9)
    sig_buy_sell = ... # convert sig_ema9 to bool ndarray

    trade_events = generate_long_short_trade_events(candles)
    report = account_report()