from .btest_mcore import Candle, Strategy
from .indicators import HMA, SMA, TREND_DOWN, TREND_UP, HalfTrend


class FalseSignalStrategy(Strategy):
    
    def __init__(self):
        super().__init__()

        self.sma = SMA(size=150)
        self.hma = HMA(size=50)
        self.hf = HalfTrend()
        
        self.tick = 0

    async def on_candle(self, candle: Candle):
        current_sma = self.sma.calculate(candle.open_time, float(candle.close))
        current_hma = self.hma.calculate(candle.open_time, float(candle.close))
        
        current_hf_signal = self.hf.calculate(candle)

        if current_sma is None:
            return
        
        if self.has_position:
            self.tick += 1
            
            if self.tick == 12:
                await self.close_position()
                self.tick = 0
            
        else:
            if current_hma > current_sma and min(candle.open, candle.close) > current_hma and current_hf_signal == TREND_UP:
                await self.open_position()
