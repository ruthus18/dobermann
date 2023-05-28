import datetime as dt
from decimal import Decimal

from .account import AccountReport
from .core import Candle, TradeAction, TradeDirection, TradeEvent
from .indicators import Indicator, _V_out


def feed(candles: list[Candle], indicator: Indicator[float, _V_out]) -> dict[dt.datetime, _V_out | None]:
    signals = {
        candle['open_time']: indicator.calculate(candle['close'])
        for candle in candles
    }
    indicator.reset()
    return signals


def backtest(
    candles: list[Candle],
    indicator: Indicator[float, bool],
    *,
    direction: TradeDirection | None = None,
) -> AccountReport:
    """Calculate trading statistics, based on series of candles, which feeded into boolean indicator

    [WARNING] Indicator in this case serving as emitter of open-close order signals
    """
    trade_events = _calculate_trade_events(candles, indicator)

    if direction:
        trade_events = [e for e in trade_events if e['direction'] == direction]

    return AccountReport(trade_events, trading_start_at=candles[0]['open_time'])


def _calculate_trade_events(candles: list[Candle], indicator: Indicator[float, bool]) -> list[TradeEvent]:
    # Backtester would not work on indicators with duplicating values (e.g. (1, 0, 0, ...)), only switching 0 <-> 1
    __id = 0

    def get_id() -> int:
        nonlocal __id
        __id += 1
        return __id

    current_id = None
    trade_events = []

    for candle in candles:
        value = indicator.calculate(candle['close'])

        if value is True:
            if current_id:
                trade_events.append(
                    TradeEvent(
                        trade_id=current_id,
                        direction=TradeDirection.BEAR,
                        size=Decimal(1),
                        time=candle['open_time'],
                        action=TradeAction.CLOSE,
                        price=candle['close'],
                    )
                )

            current_id = get_id()
            trade_events.append(
                TradeEvent(
                    trade_id=current_id,
                    direction=TradeDirection.BULL,
                    size=Decimal(1),
                    time=candle['open_time'],  # IRL this is (open_time + timeframe.timedelta)
                    action=TradeAction.OPEN,
                    price=candle['close'],  # Also need to consider time lag (strategy latency, network latency)
                                            # and round-trip transaction cost
                )
            )

        elif value is False:
            if current_id:
                trade_events.append(
                    TradeEvent(
                        trade_id=current_id,
                        direction=TradeDirection.BULL,
                        size=Decimal(1),
                        time=candle['open_time'],
                        action=TradeAction.CLOSE,
                        price=candle['close'],
                    )
                )

            current_id = get_id()
            trade_events.append(
                TradeEvent(
                    trade_id=current_id,
                    direction=TradeDirection.BEAR,
                    size=Decimal(1),
                    time=candle['open_time'],
                    action=TradeAction.OPEN,
                    price=candle['close'],
                )
            )

    indicator.reset()

    if trade_events[-1]['action'] == TradeAction.OPEN:
        trade_events.pop()

    return trade_events
