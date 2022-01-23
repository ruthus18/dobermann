import enum

from dobermann import Candle, PositionType, Strategy, Timeframe, indicators
from dobermann.indicators import EMACross


class BollingerTestStrategy(Strategy):

    def __init__(self):
        super().__init__()
        self.ind_bollinger = indicators.BollingerBands()

    def on_candle(self, candle: Candle, _):
        lower_band, sma, upper_band = self.ind_bollinger.calculate(candle)
        if sma is None:
            return

        price = candle.close

        if self.active_position is None:
            if price < lower_band:
                self.open_market_position(PositionType.LONG)

            elif price > upper_band:
                self.open_market_position(PositionType.SHORT)

        elif self.active_position.type == PositionType.LONG and price > sma:
            self.close_market_position()

        elif self.active_position.type == PositionType.SHORT and price < sma:
            self.close_market_position()


class CrossEMAStrategy(Strategy):

    def __init__(self):
        super().__init__()
        self.cross_ema = EMACross(short_ema_size=50, long_ema_size=100)

    def on_candle(self, candle: Candle, _):
        signal = self.cross_ema.calculate(candle)

        if signal is None or signal == EMACross.Signal.NEUTRAL:
            return
        
        if self.active_position:
            self.close_market_position()

        if signal == EMACross.Signal.BEAR:
            self.open_market_position(PositionType.LONG)

        elif signal == EMACross.Signal.BULL:
            self.open_market_position(PositionType.SHORT)


class FractalEMAStrategy(Strategy):

    class Action(enum.IntEnum):
        SELL = -1
        DO_NOTHING = 0
        BUY = 1

    def __init__(self):
        super().__init__()

        self.signal_5m = None
        self.signal_1h = None

        self.ema_cross_1h = EMACross(short_ema_size=20, long_ema_size=50)
        self.ema_cross_5m = EMACross(short_ema_size=20, long_ema_size=50)


    def on_candle(self, candle: Candle, timeframe: Timeframe):  # noqa
        if timeframe == Timeframe.H1:
            self.signal_1h = self.ema_cross_1h.calculate(candle)

        elif timeframe == Timeframe.M5:
            self.signal_5m = self.ema_cross_5m.calculate(candle)

        else:
            raise RuntimeError('Wrong timeframe, only (5m, 1h)')

        if self.signal_1h == EMACross.Signal.BULL and self.signal_5m == EMACross.Signal.BULL:
            self.open_market_position(PositionType.LONG)

        elif self.signal_1h == EMACross.Signal.BEAR and self.signal_5m == EMACross.Signal.BEAR:
            self.open_market_position(PositionType.SHORT)

        elif self.active_position and self.signal_5m != EMACross.Signal.NEUTRAL:
            self.close_market_position()
