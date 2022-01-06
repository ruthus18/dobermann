from dobermann import Candle, PositionType, Strategy, indicators


class BollingerTestStrategy(Strategy):

    def __init__(self):
        super().__init__()

        self.ind_bollinger = indicators.BollingerBands()

    def on_candle(self, candle: Candle):
        lower_band, sma, upper_band = self.ind_bollinger.calculate(candle)
        if sma is None:
            return

        price = candle.close

        if self.exchange.active_position is None:
            if price < lower_band:
                self.exchange.open_market_position(PositionType.LONG)

            elif price > upper_band:
                self.exchange.open_market_position(PositionType.SHORT)

        elif self.exchange.active_position.type == PositionType.LONG and price > sma:
            self.exchange.close_market_position()

        elif self.exchange.active_position.type == PositionType.SHORT and price < sma:
            self.exchange.close_market_position()
