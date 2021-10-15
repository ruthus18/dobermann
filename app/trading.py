import logging
import typing as tp
from abc import ABC, abstractmethod
from decimal import Decimal

import numpy as np
import pandas as pd

from .binance_client import Candle

logger = logging.getLogger(__name__)


def RoundedDecimal(value: tp.Any, prec: int) -> Decimal:
    return round(Decimal(value), prec)


def to_candles_df(candles_list: tp.List[Candle]) -> pd.DataFrame:
    return pd.DataFrame(c.dict() for c in candles_list).sort_values('open_time').set_index('open_time')


class Indicator(ABC):
    min_candles_size = NotImplemented

    def __init__(self, init_candles: tp.List[Candle]):
        self.candles_df = to_candles_df(init_candles)

    @abstractmethod
    def _calc(self) -> tp.Any: ...

    def process_new_candle(self, candle: Candle) -> tp.Any:
        self.candles_df = pd.concat([
            self.candles_df, to_candles_df([candle])
        ])
        if len(self.candles_df) < self.min_candles_size:
            logging.warning(f'Not enough candles data: {len(self.candles_df)}/{self.sma_size}')
            return

        return self._calc()


class BollingerBandsIndicator(Indicator):

    def __init__(self, init_candles: tp.List[Candle], sma_size: int = 20, stdev_size: int = 2):
        super().__init__(init_candles)

        self.sma_size = sma_size
        self.stdev_size = stdev_size

        self.min_candles_size = sma_size

        self.s_sma = pd.Series(dtype=object)
        self.s_stdev = pd.Series(dtype=object)
        self.s_lower_band = pd.Series(dtype=object)
        self.s_upper_band = pd.Series(dtype=object)

    def _calc(self) -> tp.Optional[tp.Tuple[Decimal, Decimal, Decimal]]:
        open_time = self.candles_df.index[-1]
        if open_time in self.s_sma:
            logging.warning('Signal already exists')
            return

        sma = RoundedDecimal(self.candles_df.close.tail(self.sma_size).mean(), 6)
        stdev = RoundedDecimal(self.candles_df.close.tail(self.sma_size).astype(np.float64).std(), 6)
        lower_band = sma - (stdev * self.stdev_size)
        upper_band = sma + (stdev * self.stdev_size)

        self.s_sma[open_time] = sma
        self.s_stdev[open_time] = stdev
        self.s_lower_band[open_time] = lower_band
        self.s_upper_band[open_time] = upper_band

        return lower_band, sma, upper_band
