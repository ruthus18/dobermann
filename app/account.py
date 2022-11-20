import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from statistics import geometric_mean

import altair as alt
import pandas as pd

from . import charts, config
from .core import TradeAction, TradeDirection, TradeEvent, TradeID

FEE = config.BYBIT_TAKER_DERIVATIVES_FEE

Decimal6 = lambda x: round(Decimal(x), 6)


@dataclass
class Trade:
    size: Decimal
    direction: TradeDirection
    time_open: dt.datetime
    time_close: dt.datetime
    price_open: float
    price_close: float

    @property
    def profit_ratio(self) -> Decimal:
        open = Decimal(self.price_open)
        close = Decimal(self.price_close)

        if self.direction == TradeDirection.BULL:
            profit = (close * (1 - FEE)) / (open * (1 + FEE))

        elif self.direction == TradeDirection.BEAR:
            profit = (open * (1 - FEE)) / (close * (1 + FEE))

        else:
            raise RuntimeError

        return round(profit, 6)

    def dict(self) -> dict:
        data = self.__dict__
        data['profit_ratio'] = self.profit_ratio
        return data


class AccountReport:
    def __init__(
        self,
        trade_events: list[TradeEvent],
        *,
        trading_start_at: dt.datetime,
        initial_equity: Decimal = Decimal(1000),
    ):
        self.trade_events = trade_events
        self.trading_start_at = trading_start_at
        self.initial_equity = initial_equity

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (profit={self.profit_ratio})'

    @cached_property
    def trades(self) -> dict[TradeID, Trade]:
        trades = {}
        # sorting: first open event, then close
        events = sorted(self.trade_events, key=lambda e: (e['trade_id'], e['action'] == TradeAction.CLOSE))

        for event in events:
            trade_id = event['trade_id']

            if event['action'] == TradeAction.OPEN:
                assert trade_id not in trades

                trades[trade_id] = Trade(
                    time_open=event['time'],
                    price_open=event['price'],
                    direction=event['direction'],
                    size=event['size'],
                    time_close=None,  # type: ignore
                    price_close=None,  # type: ignore
                )

            elif event['action'] == TradeAction.CLOSE:
                trades[trade_id].time_close = event['time']
                trades[trade_id].price_close = event['price']

            else:
                raise RuntimeError

        return trades

    @cached_property
    def equities(self) -> dict[dt.datetime, Decimal]:
        current_equity = self.initial_equity
        equities = {self.trading_start_at: current_equity}

        for event in sorted(self.trade_events, key=lambda e: e['time']):
            trade = self.trades[event['trade_id']]

            if event['action'] == TradeAction.OPEN:
                equities[trade.time_open] = current_equity

            elif event['action'] == TradeAction.CLOSE:
                current_equity *= (trade.profit_ratio * trade.size)
                equities[trade.time_close] = round(current_equity, 2)

            else:
                raise RuntimeError

        return equities

    @cached_property
    def leverages(self) -> dict[dt.datetime, Decimal]:
        current_leverage = Decimal(0)
        leverages = {self.trading_start_at: current_leverage}

        for event in sorted(self.trade_events, key=lambda e: e['time']):
            if event['action'] == TradeAction.OPEN:
                current_leverage += event['size']

            elif event['action'] == TradeAction.CLOSE:
                current_leverage -= event['size']

            else:
                raise RuntimeError

            leverages[event['time']] = current_leverage

        return leverages

    @cached_property
    def drawdowns(self) -> dict[dt.datetime, Decimal]:
        equities = list(self.equities.items())
        start_time, max_equity = equities[0]

        drawdowns = {start_time: Decimal(0)}

        for time, equity in equities[1:]:
            max_equity = max(equity, max_equity)
            drawdowns[time] = round(1 - (equity / max_equity), 6)

        return drawdowns

    @cached_property
    def profit_ratio(self) -> Decimal:
        return round(list(self.equities.values())[-1] / self.initial_equity, 6)

    @cached_property
    def summary(self) -> dict:
        profit_s = pd.Series(t.profit_ratio for t in self.trades.values())
        leverage_s = pd.Series(self.leverages)
        return {
            'profit_ratio': self.profit_ratio,
            'mean_profit_ratio': Decimal6(profit_s.mean()),
            'gmean_profit_ratio': Decimal6(geometric_mean(profit_s)),
            'max_drawdown': max(self.drawdowns.values()),
            'total_trades': len(profit_s),
            'success_trades': len(profit_s[profit_s > 1]),
            'fail_trades': len(profit_s[profit_s <= 1]),
            'success_trades_ratio': Decimal6(len(profit_s[profit_s > 1]) / len(profit_s)),
            'avg_leverage_used': Decimal6(leverage_s[leverage_s != 0].mean()),
            'max_leverage_used': leverage_s.max(),
            # sharpe ratio = np.sqrt(days in y) * np.mean(net_returns) / np.std(net_returns)
        }

    @property
    def chart(self) -> alt.VConcatChart:
        return charts.equity_leverage_drawdown_chart(self.equities, self.leverages, self.drawdowns)

    @property
    def equity_chart(self) -> alt.LayerChart:
        return charts.equity_chart(self.equities)

    @property
    def leverage_chart(self) -> alt.LayerChart:
        return charts.leverage_chart(self.leverages)

    @property
    def drawdown_chart(self) -> alt.LayerChart:
        return charts.drawdown_chart(self.drawdowns)
