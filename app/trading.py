import datetime as dt
import logging
import typing as tp
from abc import ABC, abstractmethod
from decimal import Decimal

import numpy as np
import pandas as pd

from .binance_client import Candle

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

    def __init__(self, sma_size: int = 20, stdev_size: int = 2):
        self.sma_size = sma_size
        self.stdev_size = stdev_size

        self.s_price = pd.Series(dtype=object)

        self.s_sma = pd.Series(dtype=object)
        self.s_stdev = pd.Series(dtype=object)
        self.s_lower_band = pd.Series(dtype=object)
        self.s_upper_band = pd.Series(dtype=object)

    def calculate(self, candle: Candle) -> tp.Optional[tp.Tuple[Decimal, Decimal, Decimal]]:
        open_time = candle.open_time
        self.s_price[open_time] = candle.close

        if len(self.s_price) < self.sma_size:
            return

        sma = RoundedDecimal(self.s_price.tail(self.sma_size).mean())
        stdev = RoundedDecimal(self.s_price.tail(self.sma_size).astype(np.float64).std())
        lower_band = sma - (stdev * self.stdev_size)
        upper_band = sma + (stdev * self.stdev_size)

        self.s_sma[open_time] = sma
        self.s_stdev[open_time] = stdev
        self.s_lower_band[open_time] = lower_band
        self.s_upper_band[open_time] = upper_band

        return lower_band, sma, upper_band


class MACDIndicator(Indicator):

    def __init__(self, ema_long_size: int = 26, ema_short_size: int = 12, ema_signal_size: int = 9):
        self.ema_long = EMA(ema_long_size)
        self.ema_short = EMA(ema_short_size)
        self.ema_signal = EMA(ema_signal_size)

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
