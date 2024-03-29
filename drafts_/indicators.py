import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Indicator(ABC):

    @abstractmethod
    def calculate(self, value: float) -> float: ...


class SMA(Indicator):
    """Simple Moving Average"""

    def __init__(self, size: int):
        self.size = size
        self._values: list[float] = []

    def calculate(self, value: float) -> float | None:
        self._values.append(value)

        if len(self._values) < self.size:
            return None

        sma = sum(self._values) / self.size
        self._values = self._values[1:]

        return sma


class EMA(Indicator):
    """Exponential Moving Average"""

    def __init__(self, size: int):
        self.size = size
        self.multiplier = round(2 / (self.size + 1), 4)

        self._values: list[float] = []
        self._last_ema: float = None

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


class WMA(Indicator):
    """Weighted Moving Average"""

    def __init__(self, size: int):
        self.size = size

        self._values: list[float] = []
        self.weights = np.arange(1, size + 1)

    def calculate(self, value: float) -> float | None:
        self._values.append(value)

        if len(self._values) < self.size:
            return None

        elif len(self._values) > self.size:
            self._values.pop(0)

        current_wma = np.average(self._values, weights=self.weights)
        return current_wma


# FIXME: DEPRECATED
# class __HMA:
#     """Hull Moving Average"""

#     def __init__(self, size: int):
#         self.size_long = size
#         self.size_short = round(size / 2)
#         self.size_smooth = round(math.sqrt(size))

#         self.s_values = pd.Series(dtype='float64')
#         self.s_raw = pd.Series(dtype='float64')
#         self.s = pd.Series(dtype='float64')

#         self.weights_long = np.arange(1, self.size_long + 1)
#         self.weights_short = np.arange(1, self.size_short + 1)
#         self.weights_smooth = np.arange(1, self.size_smooth + 1)

#     def calculate(self, time: dt.datetime, value: float) -> float | None:
#         self.s_values[time] = value

#         if len(self.s_values) < self.size_long:
#             return None

#         wma_long = np.average(self.s_values.tail(self.size_long), weights=self.weights_long)
#         wma_short = np.average(self.s_values.tail(self.size_short), weights=self.weights_short)

#         raw_hma = (2 * wma_short) - wma_long
#         self.s_raw[time] = raw_hma

#         if len(self.s_raw) < self.size_smooth:
#             return None

#         current_hma = np.average(self.s_raw.tail(self.size_smooth), weights=self.weights_smooth)
#         self.s[time] = current_hma

#         return current_hma


# # FIXME: DEPRECATED
# class __ATR:
#     """Average True Range"""
#     def __init__(self, size: int):
#         self.size = size

#         self.prev_candle: dict = None
#         self.s_true_range = pd.Series(dtype='float64')
#         self.s = pd.Series(dtype='float64')

#     def calculate(self, candle: Candle) -> float | None:
#         if not self.prev_candle:
#             self.prev_candle = candle
#             return None

#         true_range = float(max(
#             (candle.high - candle.low),
#             abs(candle.high - self.prev_candle.close),
#             abs(candle.low - self.prev_candle.close)
#         ))
#         self.prev_candle = candle

#         self.s_true_range[candle.open_time] = true_range

#         if len(self.s_true_range) < self.size:
#             return None

#         current_atr = np.average(self.s_true_range.tail(self.size))
#         self.s[candle.open_time] = current_atr

#         return current_atr


# TREND_UP = 1
# TREND_DOWN = -1


# FIXME: DEPRECATED
# class __HalfTrend:
#     """HalfTrend indicator

#     PineScript Implementation: https://tradingview.com/script/U1SJ8ubc-HalfTrend/
#     """
#     def __init__(self, amplitude: int = 2, channel_deviation: int = 2):
#         self.amplitude = amplitude
#         self.channel_deviation = channel_deviation

#         self.atr = ATR(100)
#         self.prev_candle: dict = None
#         self.trend = TREND_UP

#         # Uptrend data
#         self.high_sma = SMA(size=self.amplitude)
#         self.max_low_price = float('-inf')

#         # Downtrend data
#         self.low_sma = SMA(size=self.amplitude)
#         self.min_high_price = float('inf')

#         self.s = pd.Series(dtype='int')

#     def calculate(self, candle: Candle) -> tp.Tuple[float | None, float | None, float | None, float | None]:
#         signal = self._calculate(candle)

#         if not MEMORY_EFFICIENT:
#             if signal is None:
#                 self.s[candle.open_time] = 0
#             else:
#                 self.s[candle.open_time] = signal

#         self.prev_candle = candle
#         return signal

#     def _calculate(self, candle: Candle) -> tp.Tuple[float | None, float | None, float | None, float | None]:
#         time = candle.open_time

#         current_low_sma = self.low_sma.calculate(time, candle.low)
#         current_high_sma = self.high_sma.calculate(time, candle.high)

#         if not current_low_sma:
#             return None

#         # TODO: Нужно разобраться, для чего нужны эти условия вместо candle['high'/'low']
#         local_high = max(candle.high, self.prev_candle.high)
#         local_low = min(candle.low, self.prev_candle.low)

#         if self.trend == TREND_UP:
#             self.max_low_price = max(self.max_low_price, local_low)

#             # Check whether uptrend is end
#             if (current_high_sma < self.max_low_price) and (candle.close < self.prev_candle.low):
#                 self.trend = TREND_DOWN
#                 self.min_high_price = local_high

#                 return TREND_DOWN

#         elif self.trend == TREND_DOWN:
#             self.min_high_price = min(self.min_high_price, local_high)

#             # Check whether downtrend is end
#             if (current_low_sma > self.min_high_price) and (candle.close > self.prev_candle.high):
#                 self.trend = TREND_UP
#                 self.max_low_price = local_low

#                 return TREND_UP

#         else:
#             raise RuntimeError()

#         return None


# FIXME: Deprecated
# class __BollingerBands:

#     def __init__(self, size: int = 20, stdev_size: int = 2):
#         self.size = size
#         self.stdev_size = stdev_size

#         self.s_price = pd.Series(dtype=object)

#         self.s_sma = pd.Series(dtype=object)
#         self.s_stdev = pd.Series(dtype=object)
#         self.s_lower_band = pd.Series(dtype=object)
#         self.s_upper_band = pd.Series(dtype=object)

#     def calculate(self, candle: Candle) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
#         open_time = candle.open_time
#         self.s_price[open_time] = candle.close

#         if len(self.s_price) < self.size:
#             return None, None, None

#         sma = Decimal(self.s_price.tail(self.size).mean())
#         stdev = Decimal(self.s_price.tail(self.size).astype(np.float64).std())
#         lower_band = sma - (stdev * self.stdev_size)
#         upper_band = sma + (stdev * self.stdev_size)

#         self.s_sma[open_time] = sma
#         self.s_stdev[open_time] = stdev
#         self.s_lower_band[open_time] = lower_band
#         self.s_upper_band[open_time] = upper_band

#         return lower_band, sma, upper_band


# FIXME: Deprecated
# class BollingerBandsEMA(Indicator):

#     def __init__(self, ema_size: int = 20, stdev_size: int = 2):
#         self.ema_size = ema_size
#         self.stdev_size = stdev_size

#         self.s_price = pd.Series(dtype=object)

#         self.ema = EMA(ema_size)
#         self.s_stdev = pd.Series(dtype=object)
#         self.s_lower_band = pd.Series(dtype=object)
#         self.s_upper_band = pd.Series(dtype=object)

#     @property
#     def s_ema(self) -> pd.Series: return self.ema.s

#     def calculate(self, candle: Candle) -> tp.Tuple[OptDecimal, OptDecimal, OptDecimal]:
#         open_time = candle.open_time
#         self.s_price[open_time] = candle.close

#         ema = self.ema.calculate(open_time, candle.close)
#         if ema is None:
#             return None, None, None

#         stdev = RoundedDecimal(self.s_price.tail(self.ema_window).astype(np.float64).std())
#         lower_band = ema - (stdev * self.stdev_size)
#         upper_band = ema + (stdev * self.stdev_size)

#         self.s_ema[open_time] = ema
#         self.s_stdev[open_time] = stdev
#         self.s_lower_band[open_time] = lower_band
#         self.s_upper_band[open_time] = upper_band

#         return lower_band, ema, upper_band


# FIXME: Deprecated
# class MACD(Indicator):

#     def __init__(self, long_ema_size: int = 26, short_ema_size: int = 12, signal_ema_size: int = 9):
#         self.long_ema = EMA(long_ema_size)
#         self.short_ema = EMA(short_ema_size)
#         self.signal_ema = EMA(signal_ema_size)

#         self.s_macd = pd.Series(dtype=object)
#         self.s_macd_histogram = pd.Series(dtype=object)

#     @property
#     def s_long_ema(self) -> pd.Series: return self.long_ema.s

#     @property
#     def s_short_ema(self) -> pd.Series: return self.short_ema.s

#     @property
#     def s_signal_ema(self) -> pd.Series: return self.signal_ema.s

#     def calculate(self, candle: Candle) -> tp.Tuple[OptDecimal, OptDecimal]:
#         time, price = candle.open_time, candle.close

#         long_value = self.long_ema.calculate(time, price)
#         short_value = self.short_ema.calculate(time, price)

#         if short_value is None or long_value is None:
#             return None, None

#         macd_value = short_value - long_value
#         self.s_macd[time] = macd_value

#         signal_value = self.ema_signal.calculate(time, macd_value)
#         if signal_value is None:
#             return None, None

#         histogram_value = macd_value - signal_value
#         self.s_macd_histogram[time] = histogram_value

#         return signal_value, histogram_value


# FIXME: Deprecated
# class EMACross(Indicator):

#     class Signal(enum.IntEnum):
#         BEAR = -1
#         NEUTRAL = 0
#         BULL = 1

#     def __init__(self, short_ema_size: int, long_ema_size: int):
#         self.long_ema = EMA(long_ema_size)
#         self.short_ema = EMA(short_ema_size)

#         self.s_signal = pd.Series(dtype=np.int8)

#     @property
#     def s_long_ema(self) -> pd.Series: return self.long_ema.s

#     @property
#     def s_short_ema(self) -> pd.Series: return self.short_ema.s

#     def calculate(self, candle: Candle) -> tp.Optional['Signal']:
#         time, price = candle.open_time, candle.close

#         long_now = self.long_ema.calculate(time, price)
#         short_now = self.short_ema.calculate(time, price)

#         if long_now is None or short_now is None or len(self.s_long_ema) < 2 or len(self.s_short_ema) < 2:
#             return None  # индикаторы не прогрелись

#         prev_diff = self.s_short_ema[-2] - self.s_long_ema[-2]
#         now_diff = short_now - long_now

#         # Короткая EMA пробивает вверх длинную EMA
#         if prev_diff > 0 and now_diff < 0:
#             signal = self.Signal.BULL

#         # Короткая EMA пробивает вниз длинную EMA
#         elif prev_diff < 0 and now_diff > 0:
#             signal = self.Signal.BEAR

#         else:
#             signal = self.Signal.NEUTRAL

#         if CACHE_SERIES:
#             self.s_signal[time] = signal

#         return signal


# FIXME: Deprecated
# class LowHighEMA(Indicator):

#     def __init__(self, ema_size: int = 20, bear_bull_size: int = 50):
#         self.ema_low = EMA(size=ema_size)
#         self.ema_high = EMA(size=ema_size)

#         self.ema_low_price = EMA(size=2)
#         self.ema_high_price = EMA(size=2)

#         # self.bear_bull_size = bear_bull_size

#         self.s_ema_bull_signal = pd.Series(dtype=np.float64)
#         self.s_ema_bear_signal = pd.Series(dtype=np.float64)

#     @property
#     def s_ema_diff_signal(self) -> pd.Series:
#         return self.s_ema_high_signal - self.s_ema_low_signal

#     def calculate(self, candle: Candle) -> tp.Any:
#         low_price, high_price = candle.low, candle.high
#         time = candle.open_time

#         # Строим канал скользящей средней
#         ema_low_value = self.ema_low.calculate(time, low_price)
#         ema_high_value = self.ema_high.calculate(time, high_price)

#         # Сглаживание цены максимально короткой EMA
#         ema_low_price = self.ema_low_price.calculate(time, low_price)
#         ema_high_price = self.ema_high_price.calculate(time, high_price)

#         if ema_low_value is None or ema_high_value is None:
#             return None, None

#         ema_bull_signal = ema_high_price - ema_high_value
#         ema_bear_signal = ema_low_price - ema_low_value

#         self.s_ema_bull_signal[time] = ema_bull_signal
#         self.s_ema_bear_signal[time] = ema_bear_signal

#         return ema_bull_signal, ema_bear_signal


# FIXME: Deprecated
# # class StohasticOscillator(Indicator):

# #     def __init__(self):
# #         ...
