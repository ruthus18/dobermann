import dataclasses
import datetime as dt
import enum
import logging
import typing as tp
from abc import ABC, abstractmethod
from decimal import Decimal
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.asyncio import tqdm

from .binance_client import BinanceClient, Candle, Timeframe
from .graphs import get_equity_graph

logger = logging.getLogger(__name__)


def to_candles_df(candles_list: tp.List[Candle]) -> pd.DataFrame:
    return pd.DataFrame(c.dict() for c in candles_list).sort_values('open_time').set_index('open_time')


def RoundedDecimal(value: tp.Any) -> Decimal:
    return round(Decimal(value), 6)


class EMA:
    def __init__(self, size: int):
        self.size = size

        self.s_data = pd.Series(dtype=object)
        self.s = pd.Series(dtype=object)

        self.multiplier = self._calc_multiplier()

    def _calc_multiplier(self) -> Decimal:
        return round(Decimal(2) / Decimal(self.size + 1), 4)

    def calculate(self, time: dt.datetime, value: Decimal) -> tp.Optional[Decimal]:
        self.s_data[time] = value

        if len(self.s_data) < self.size:
            return

        if len(self.s) == 0:
            current_ema = RoundedDecimal(self.s_data.tail(self.size).mean())
        else:
            current_ema = (value * self.multiplier) + (self.s[-1] * (1 - self.multiplier))

        self.s[time] = current_ema
        return current_ema


# TODO: Валидировать временной ряд на наличие разрывов (подключать для live-трейдинга как предохранитель)
class Indicator(ABC):

    @abstractmethod
    def calculate(self, candle: Candle) -> tp.Any: ...


class BollingerBandsIndicator(Indicator):

    def __init__(self, sma_window: int = 20, stdev_size: int = 2):
        self.sma_window = sma_window
        self.stdev_size = stdev_size

        self.s_price = pd.Series(dtype=object)

        self.s_sma = pd.Series(dtype=object)
        self.s_stdev = pd.Series(dtype=object)
        self.s_lower_band = pd.Series(dtype=object)
        self.s_upper_band = pd.Series(dtype=object)

    def calculate(self, candle: Candle) -> tp.Optional[tp.Tuple[Decimal, Decimal, Decimal]]:
        open_time = candle.open_time
        self.s_price[open_time] = candle.close

        if len(self.s_price) < self.sma_window:
            return

        sma = RoundedDecimal(self.s_price.tail(self.sma_window).mean())
        stdev = RoundedDecimal(self.s_price.tail(self.sma_window).astype(np.float64).std())
        lower_band = sma - (stdev * self.stdev_size)
        upper_band = sma + (stdev * self.stdev_size)

        self.s_sma[open_time] = sma
        self.s_stdev[open_time] = stdev
        self.s_lower_band[open_time] = lower_band
        self.s_upper_band[open_time] = upper_band

        return lower_band, sma, upper_band


class BollingerBandsEMAIndicator(Indicator):

    def __init__(self, ema_window: int = 20, stdev_size: int = 2):
        self.ema_window = ema_window
        self.stdev_size = stdev_size

        self.s_price = pd.Series(dtype=object)

        self.ema = EMA(ema_window)
        self.s_stdev = pd.Series(dtype=object)
        self.s_lower_band = pd.Series(dtype=object)
        self.s_upper_band = pd.Series(dtype=object)

    def calculate(self, candle: Candle) -> tp.Optional[tp.Tuple[Decimal, Decimal, Decimal]]:
        open_time = candle.open_time
        self.s_price[open_time] = candle.close

        ema = self.ema.calculate(open_time, candle.close)
        if ema is None:
            return

        stdev = RoundedDecimal(self.s_price.tail(self.ema_window).astype(np.float64).std())
        lower_band = ema - (stdev * self.stdev_size)
        upper_band = ema + (stdev * self.stdev_size)

        self.s_ema[open_time] = ema
        self.s_stdev[open_time] = stdev
        self.s_lower_band[open_time] = lower_band
        self.s_upper_band[open_time] = upper_band

        return lower_band, ema, upper_band

    @property
    def s_ema(self) -> pd.Series: return self.ema.s


class MACDIndicator(Indicator):

    def __init__(self, ema_long_window: int = 26, ema_short_window: int = 12, ema_signal_window: int = 9):
        self.ema_long = EMA(ema_long_window)
        self.ema_short = EMA(ema_short_window)
        self.ema_signal = EMA(ema_signal_window)

        self.s_macd = pd.Series(dtype=object)
        self.s_macd_histogram = pd.Series(dtype=object)

    def calculate(self, candle: Candle) -> tp.Optional[tp.Tuple[Decimal, Decimal]]:
        open_time = candle.open_time
        price = candle.close

        ema_short = self.ema_short.calculate(open_time, price)
        ema_long = self.ema_long.calculate(open_time, price)

        if ema_short is None or ema_long is None:
            return

        macd = ema_short - ema_long
        self.s_macd[open_time] = macd

        ema_signal = self.ema_signal.calculate(open_time, macd)
        if ema_signal is None:
            return

        macd_histogram = macd - ema_signal
        self.s_macd_histogram[open_time] = macd_histogram

        return ema_signal, macd_histogram

    @property
    def s_ema_long(self) -> pd.Series: return self.ema_long.s

    @property
    def s_ema_short(self) -> pd.Series: return self.ema_short.s

    @property
    def s_ema_signal(self) -> pd.Series: return self.ema_signal.s


BINANCE_F_COMISSION = Decimal(0.0004)


class PositionType(str, enum.Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'

    def __str__(self):
        return self.value


@dataclasses.dataclass
class Position:
    type: PositionType
    enter_time: dt.datetime
    enter_price: Decimal
    exit_time: dt.datetime = None
    exit_price: Decimal = None

    @cached_property
    def profit(self) -> Decimal:
        C = BINANCE_F_COMISSION

        if self.type == PositionType.LONG:
            profit = ((1 - C) * self.exit_price) - ((1 + C) * self.enter_price)
        else:
            profit = ((1 - C) * self.enter_price) - ((1 + C) * self.exit_price)

        return RoundedDecimal(profit)

    @cached_property
    def profit_ratio(self) -> Decimal:
        C = BINANCE_F_COMISSION

        if self.type == PositionType.LONG:
            ratio = ((1 - C) * self.exit_price) / ((1 + C) * self.enter_price)
        else:
            ratio = ((1 - C) * self.enter_price) / ((1 + C) * self.exit_price)

        return RoundedDecimal(ratio)

    def as_dict(self) -> dict:
        return {
            'type': self.type,
            'enter_time': self.enter_time,
            'enter_price': self.enter_price,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'profit': self.profit,
            'profit_ratio': self.profit_ratio,
        }


class Strategy(ABC):

    def __init__(self):
        self.positions: tp.List[Position] = []
        self.active_position: tp.Optional[Position] = None

    def open_position(
        self,
        enter_time: dt.datetime,
        enter_price: Decimal,
        position_type: PositionType = PositionType.LONG,
    ):
        if self.active_position is not None:
            raise RuntimeError('Position already opened')

        self.active_position = Position(
            type=position_type,
            enter_time=enter_time,
            enter_price=enter_price,
        )

    def close_position(self, exit_time: dt.datetime, exit_price: Decimal):
        if self.active_position is None:
            raise RuntimeError("Position wasn't open")

        self.active_position.exit_time = exit_time
        self.active_position.exit_price = exit_price

        self.positions.append(self.active_position)
        self.active_position = None

    @abstractmethod
    def on_candle(self, candle: Candle): ...


class BollingerTestStrategy(Strategy):

    def __init__(self):
        super().__init__()

        self.ind_bollinger = BollingerBandsIndicator()

    def on_candle(self, candle: Candle):
        ind_result = self.ind_bollinger.calculate(candle)
        if ind_result is None:
            return

        time, price = candle.open_time, candle.close
        lower_band, sma, upper_band = ind_result

        if self.active_position is None:
            if price < lower_band:
                self.open_position(time, price, PositionType.LONG)

            elif price > upper_band:
                self.open_position(time, price, PositionType.SHORT)

        elif self.active_position.type == PositionType.LONG and price > sma:
            self.close_position(time, price)

        elif self.active_position.type == PositionType.SHORT and price < sma:
            self.close_position(time, price)


# TODO: Stop Loss/Take Profit check with 1m candles
# TODO: Clear strategy on second run
class Backtester:

    def __init__(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_at: dt.datetime,
        end_at: dt.datetime,
        strategy: Strategy
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_at = start_at
        self.end_at = end_at

        self.strategy = strategy
        self._candles: tp.Optional[tp.List[Candle]] = None

        self.equities = pd.Series(dtype=object)

    async def run(self):
        await self._fetch_candles()

        logger.info('Perform strategy...')
        for candle in tqdm(self._candles):
            self.strategy.on_candle(candle)

        logger.info('Done!')

    async def _fetch_candles(self):
        api_client = await BinanceClient.init()

        logger.info('Fetching candles...')
        self._candles = [
            candle async for candle in api_client.get_futures_historical_candles(
                symbol=self.symbol, timeframe=self.timeframe, start=self.start_at, end=self.end_at
            )
        ]
        await api_client.close()

    def _calculate_equity(self, initial_amount: Decimal):
        equity = initial_amount
        self.equities[self.start_at] = equity

        for position in self.strategy.positions:
            equity *= position.profit_ratio
            self.equities[position.exit_time] = round(equity, 1)

    @property
    def equity_graph(self, initial_amount: Decimal = Decimal(10000)) -> go.Figure:
        if len(self.equities) == 0:
            self._calculate_equity(initial_amount)

        return get_equity_graph(
            go.Scatter(x=self.equities.index, y=self.equities.values, name='Equity', line=dict(color='#7658e0'))
        )

    @property
    def summary(self):
        return {
            'max_dropdown': min(pos.profit_ratio for pos in self.strategy.positions)
        }
