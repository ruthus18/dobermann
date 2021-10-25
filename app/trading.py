import datetime as dt
import enum
import logging
import typing as tp
from abc import ABC, abstractmethod
from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.asyncio import tqdm

from .binance_client import BinanceClient, Candle
from .binance_client import OrderSide as OrderType
from .binance_client import Timeframe
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


class Strategy(ABC):

    def __init__(self):
        self.orders = pd.Series(dtype=object)

    @abstractmethod
    def on_candle(self, candle: Candle): ...


class PositionType(str, enum.Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'


class BollingerTestStrategy(Strategy):

    def __init__(self):
        self.ind_bollinger = BollingerBandsIndicator()

        self.orders = []
        self.current_position: tp.Optional[PositionType] = None

    def open_position(self, time: dt.datetime, price: Decimal, position_type: PositionType = PositionType.LONG):
        if self.current_position is not None:
            raise RuntimeError('Position already opened')

        self.orders.append({
            'time': time,
            'price': price,
            'order_type': OrderType.BUY if position_type == PositionType.LONG else OrderType.SELL,
        })
        self.current_position = position_type

    def close_position(self, time: dt.datetime, price: Decimal):
        if self.current_position is None:
            raise RuntimeError("Position wasn't open")

        self.orders.append({
            'time': time,
            'price': price,
            'order_type': OrderType.SELL if self.current_position == PositionType.LONG else OrderType.BUY,
        })
        self.current_position = None

    def on_candle(self, candle: Candle):
        ind_result = self.ind_bollinger.calculate(candle)
        if ind_result is None:
            return

        time, price = candle.open_time, candle.close
        lower_band, sma, upper_band = ind_result

        if self.current_position is None:
            if price < lower_band:
                self.open_position(time, price, PositionType.LONG)

            elif price > upper_band:
                self.open_position(time, price, PositionType.SHORT)

        elif self.current_position == PositionType.LONG and price > sma:
            self.close_position(time, price)

        elif self.current_position == PositionType.SHORT and price < sma:
            self.close_position(time, price)


BINANCE_F_COMISSION = Decimal(0.0004)


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
        self._orders = None
        self.positions = None
        self.equities = None

        self._candles: tp.Optional[tp.AsyncGenerator[None, Candle]] = None

    async def _fetch_candles(self):
        api_client = await BinanceClient.init()

        self._candles = [
            candle async for candle in api_client.get_futures_historical_candles(
                symbol=self.symbol, timeframe=self.timeframe, start=self.start_at, end=self.end_at
            )
        ]
        await api_client.close()

    async def run(self):
        await self._fetch_candles()

        for candle in tqdm(self._candles):
            self.strategy.on_candle(candle)

        self._calculate_positions()

    # TODO: Comissions
    def _calculate_positions(self):
        self.positions = []

        for i in range(0, len(self.strategy.orders) // 2 * 2, 2):
            enter_order = self.strategy.orders[i]
            exit_order = self.strategy.orders[i+1]

            position = {
                'enter_time': enter_order['time'],
                'exit_time': exit_order['time'],
                'enter_price': enter_order['price'],
                'exit_price': exit_order['price'],
            }

            if enter_order['order_type'] == OrderType.BUY:
                position['profit'] = exit_order['price'] - enter_order['price']
                position['profit_ratio'] = RoundedDecimal(exit_order['price'] / enter_order['price'])
            else:
                position['profit'] = enter_order['price'] - exit_order['price']
                position['profit_ratio'] = RoundedDecimal(enter_order['price'] / exit_order['price'])

            self.positions.append(position)

    def _calculate_equity(self, initial_amount: Decimal):
        equity = initial_amount
        self.equities[self.start_at] = equity

        for position in self.positions:
            equity *= position['profit_ratio']
            self.equities[position['exit_time']] = round(equity, 1)

    @property
    def equity_graph(self, initial_amount: Decimal = Decimal(10000)) -> go.Figure:
        if self.equities is None:
            self.equities = pd.Series(dtype=object)
            self._calculate_equity(initial_amount)

        return get_equity_graph(
            go.Scatter(x=self.equities.index, y=self.equities.values, name='Equity', line=dict(color='#7658e0'))
        )

    @property
    def summary(self):
        return {
            'max_dropdown': min(pos['profit_ratio'] for pos in self.positions)
        }
