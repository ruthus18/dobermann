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


def ema_multiplier(size: int) -> Decimal:
    return round(Decimal(2) / Decimal(size + 1), 4)


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


# TODO: Вынести подсчеты EMA в отдельную абстракцию
class MACDIndicator(Indicator):

    def __init__(self, ema_long_size: int = 26, ema_short_size: int = 12, ema_signal_size: int = 9):
        self.ema_long_size = ema_long_size
        self.ema_short_size = ema_short_size
        self.ema_signal_size = ema_signal_size

        self.s_price = pd.Series(dtype=object)

        self.s_ema_long = pd.Series(dtype=object)
        self.s_ema_short = pd.Series(dtype=object)
        self.s_macd = pd.Series(dtype=object)
        self.s_ema_signal = pd.Series(dtype=object)
        self.s_macd_histogram = pd.Series(dtype=object)

    def calculate(self, candle: Candle) -> tp.Optional[tp.Tuple[Decimal, Decimal]]:
        open_time = candle.open_time
        self.s_price[open_time] = candle.close

        # 1. Calc the short EMA
        if len(self.s_price) < self.ema_short_size:
            return

        if len(self.s_ema_short) == 0:
            ema_short = RoundedDecimal(self.s_price.tail(self.ema_short_size).mean())
        else:
            _ema_mult = ema_multiplier(self.ema_short_size)
            ema_short = (self.s_price[-1] * _ema_mult) + (self.s_ema_short[-1] * (1 - _ema_mult))

        self.s_ema_short[open_time] = ema_short

        # 2. Calc the long EMA
        if len(self.s_price) < self.ema_long_size:
            return

        if len(self.s_ema_long) == 0:
            ema_long = RoundedDecimal(self.s_price.tail(self.ema_long_size).mean())
        else:
            _ema_mult = ema_multiplier(self.ema_long_size)
            ema_long = (self.s_price[-1] * _ema_mult) + (self.s_ema_long[-1] * (1 - _ema_mult))

        self.s_ema_long[open_time] = ema_long

        # 3. Calc the MACD
        macd = ema_short - ema_long
        self.s_macd[open_time] = macd

        # 4. Calc the signal EMA
        if len(self.s_macd) < self.ema_signal_size:
            return

        if len(self.s_ema_signal) == 0:
            ema_signal = RoundedDecimal(self.s_macd.tail(self.ema_signal_size).mean())
        else:
            _ema_mult = ema_multiplier(self.ema_signal_size)
            ema_signal = (self.s_macd[-1] * _ema_mult) + (self.s_ema_signal[-1] * (1 - _ema_mult))

        self.s_ema_signal[open_time] = ema_signal

        # 5. Calc the MACD Histogram
        macd_histogram = macd - ema_signal
        self.s_macd_histogram[open_time] = macd_histogram

        return ema_signal, macd_histogram
