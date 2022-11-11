from abc import ABC, abstractmethod

import numpy as np


class Indicator(ABC):

    @abstractmethod
    def calculate(self, value: float) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class SMA(Indicator):
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


class EMA(Indicator):
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


class WMA(Indicator):
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
