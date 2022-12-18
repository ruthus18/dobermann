import typing as t
from abc import ABC, abstractmethod

import numpy as np

from .core import Candle, Timeframe

_V = t.TypeVar('_V')


class Indicator(ABC, t.Generic[_V]):

    @abstractmethod
    def calculate(self, value: float) -> _V | None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class SMA(Indicator[float]):
    """Simple Moving Average"""

    def __init__(self, size: int):
        self.size = size
        self._values: list[float] = []

    def __str__(self) -> str:
        return f'SMA({self.size})'

    def calculate(self, value: float) -> float | None:
        self._values.append(value)

        if len(self._values) < self.size:
            return None

        sma = sum(self._values) / self.size
        self._values = self._values[1:]

        return sma

    def reset(self) -> None:
        self._values = []


class EMA(Indicator[float]):
    """Exponential Moving Average"""

    def __init__(self, size: int):
        self.size = size
        self.multiplier = round(2 / (self.size + 1), 4)

        self._values: list[float] = []
        self._last_ema: float | None = None

    def __str__(self) -> str:
        return f'EMA({self.size})'

    def calculate(self, value: float) -> float | None:
        if self._last_ema:
            ema = (value * self.multiplier) + (self._last_ema * (1 - self.multiplier))
        else:
            self._values.append(value)
            if len(self._values) < self.size:
                return None

            ema = sum(self._values) / self.size

        self._last_ema = ema
        return ema

    def reset(self) -> None:
        self._values = []
        self._last_ema = None


class DEMA(Indicator[float]):

    def __init__(self, size: int):
        self.size = size
        self.ema = EMA(size)
        self.ema2 = EMA(size)  # smoothen EMA

    def __str__(self) -> str:
        return f'DEMA({self.size})'

    def calculate(self, value: float) -> float | None:
        ema_value = self.ema.calculate(value)
        if ema_value is None:
            return None

        ema2_value = self.ema2.calculate(ema_value)
        if ema2_value is None:
            return None

        return 2 * ema_value - ema2_value

    def reset(self) -> None:
        self.ema.reset()
        self.ema2.reset()


class TEMA(Indicator[float]):

    def __init__(self, size: int):
        self.size = size
        self.ema = EMA(size)
        self.ema2 = EMA(size)  # smoothen EMA
        self.ema3 = EMA(size)  # doube-smoothen EMA

    def __str__(self) -> str:
        return f'TEMA({self.size})'

    def calculate(self, value: float) -> float | None:
        ema_value = self.ema.calculate(value)
        if ema_value is None:
            return None

        ema2_value = self.ema2.calculate(ema_value)
        if ema2_value is None:
            return None

        ema3_value = self.ema3.calculate(ema2_value)
        if ema3_value is None:
            return None

        return 3 * ema_value - 3 * ema2_value + ema3_value

    def reset(self) -> None:
        self.ema.reset()
        self.ema2.reset()
        self.ema3.reset()


class WMA(Indicator[float]):
    """Weighted Moving Average"""

    def __init__(self, size: int):
        self.size = size

        self._values: list[float] = []
        self.weights = np.arange(1, size + 1)

    def __str__(self) -> str:
        return f'WMA({self.size})'

    def calculate(self, value: float) -> float | None:
        self._values.append(value)

        if len(self._values) < self.size:
            return None

        elif len(self._values) > self.size:
            self._values.pop(0)

        current_wma = np.average(self._values, weights=self.weights)
        return current_wma  # type: ignore

    def reset(self) -> None:
        self._values = []


MAType = SMA | EMA | DEMA | TEMA | WMA


class MACross(Indicator[bool]):
    """
    * MA(short) cross MA(long) bottom-up  = True
    * MA(short) cross MA(long) top-down   = False
    * MA no cross or not enough data      = None
    """
    def __init__(self, ma_short: MAType, ma_long: MAType):
        self.ma_short = ma_short
        self.ma_long = ma_long

        self.p_value_short: float | None = None
        self.p_value_long: float | None = None

    def __str__(self) -> str:
        return f'MACross({self.ma_short}, {self.ma_long})'

    def _calculate(self, value_short: float | None, value_long: float | None) -> bool | None:
        if not self.p_value_short or not self.p_value_long:
            return None  # Not enough data

        value_short = t.cast(float, value_short)
        value_long = t.cast(float, value_long)

        if self.p_value_short <= self.p_value_long and value_short > value_long:
            return True  # Cross bottom-up

        if self.p_value_short >= self.p_value_long and value_short < value_long:
            return False  # Cross top-down

        return None  # No cross

    def calculate(self, value: float) -> bool | None:
        value_short = self.ma_short.calculate(value)
        value_long = self.ma_long.calculate(value)

        cross = self._calculate(value_short, value_long)

        self.p_value_short = value_short
        self.p_value_long = value_long

        return cross

    def reset(self) -> None:
        self.ma_short.reset()
        self.ma_long.reset()

        self.p_value_short = None
        self.p_value_long = None


# FIXME: Абстракция потекла :(
#        (`Candle` - более сложная структура данных нежели numeric/bool и плохо ложится на текущий инструмент)
class TimeScale(Indicator[Candle]):

    def __init__(self, in_scale: Timeframe, out_scale: Timeframe):
        if out_scale.timedelta <= in_scale.timedelta:
            raise ValueError('Output scale must be greather than input scale')

        self.in_scale = in_scale
        self.out_scale = out_scale

        self.candles_mem: list[Candle] = []
        self.scale_factor = int(self.out_scale.timedelta / self.in_scale.timedelta)

    def calculate(self, value: Candle) -> Candle | None:
        self.candles_mem.append(value)

        if len(self.candles_mem) != self.scale_factor:
            return None

        scaled_candle = Candle(
            open_time=self.candles_mem[0]['open_time'],
            open=self.candles_mem[0]['open'],
            close=self.candles_mem[-1]['close'],
            low=min(c['low'] for c in self.candles_mem),
            high=max(c['high'] for c in self.candles_mem),
            volume=sum(c['volume'] for c in self.candles_mem),
        )
        self.candles_mem = []

        return scaled_candle

    def reset(self) -> None:
        self.candles_mem = []
