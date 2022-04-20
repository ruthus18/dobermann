import datetime as dt
import enum
import logging
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import cached_property
from statistics import geometric_mean

import pandas as pd
import plotly.graph_objects as go
from tqdm.asyncio import tqdm

from .config import settings

from . import graphs
from .binance_client import BinanceClient, Candle, Timeframe
from .utils import OptDecimal, RoundedDecimal

logger = logging.getLogger(__name__)

BINANCE_FUTURES_LIMIT_COMISSION = Decimal('0.0002')
BINANCE_FUTURES_MARKET_COMISSION = Decimal('0.0004')

EQUITY_INITIAL_AMOUNT = Decimal(1000)


class PositionType(str, enum.Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'

    def __str__(self):
        return self.value


@dataclass
class Position:
    type: PositionType

    enter_time: dt.datetime
    enter_price: Decimal
    enter_comission: Decimal

    exit_comission: Decimal = None
    exit_time: dt.datetime = None
    exit_price: Decimal = None

    stop_loss: Decimal = None
    take_profit: Decimal = None

    @cached_property
    def profit(self) -> Decimal:
        C_IN = self.enter_comission
        C_OUT = self.exit_comission

        if self.type == PositionType.LONG:
            profit = ((1 - C_OUT) * self.exit_price) - ((1 + C_IN) * self.enter_price)
        else:
            profit = ((1 - C_IN) * self.enter_price) - ((1 + C_OUT) * self.exit_price)

        return RoundedDecimal(profit)

    @cached_property
    def profit_ratio(self) -> Decimal:
        C_IN = self.enter_comission
        C_OUT = self.exit_comission

        if self.type == PositionType.LONG:
            ratio = ((1 - C_OUT) * self.exit_price) / ((1 + C_IN) * self.enter_price)
        else:
            ratio = ((1 - C_IN) * self.enter_price) / ((1 + C_OUT) * self.exit_price)

        return RoundedDecimal(ratio)

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            'profit': self.profit,
            'profit_ratio': self.profit_ratio,
        }


# TODO: Отразить в доке и названии, что это Exchange для backtesting-а
class Exchange:
    """Примитив для выполнения операций на бирже.
    Предоставляет набор операций для открытия/закрытия позиций, отслеживания Stop Loss/Take Profit
    для открытых позиций и сбора статистики по торговому счету.
    ВАЖНО: Вызов метода `Exchange.on_candle()` должен происходить перед вызовом метода `Strategy.on_candle()`,
    т.к. для совершения действий на бирже сперва необходимо обновить состояние биржи (обработать текущие SL/TP и пр).
    """
    def __init__(
        self,
        limit_order_comission: OptDecimal = BINANCE_FUTURES_LIMIT_COMISSION,
        market_order_comission: OptDecimal = BINANCE_FUTURES_MARKET_COMISSION,
    ):
        self.positions: tp.List[Position] = []
        self.active_position: tp.Optional[Position] = None

        self.market_order_comission = market_order_comission
        # TODO: На данный момент не используется, на этапе MVP не планируется поддержка limit order
        self.limit_order_comission = limit_order_comission

        self._current_candle: tp.Optional[Candle] = None

    def open_market_position(
        self,
        position_type: PositionType = PositionType.LONG,
        stop_loss: OptDecimal = None,
        take_profit: OptDecimal = None,
    ):
        if not self._current_candle:
            raise RuntimeError('Wrong state of exchange, you should call first `.on_candle()` method!')

        if self.active_position:
            raise RuntimeError('Position already opened')

        self.active_position = Position(
            type=position_type,
            enter_time=self._current_candle.open_time,
            enter_price=self._current_candle.close,
            enter_comission=self.market_order_comission,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _close_position(self, price: Decimal):
        self.active_position.exit_time = self._current_candle.open_time
        self.active_position.exit_price = price
        self.active_position.exit_comission = self.market_order_comission

        self.positions.append(self.active_position)
        self.active_position = None

    def close_market_position(self):
        if self.active_position is None:
            raise RuntimeError("Position wasn't open")

        # Стоимость = цене закрытия предыдущей свечи
        # (в данном случае делаем допущение, что время на получение свечи через API и генерацию сигнала = 0)
        self._close_position(price=self._current_candle.close)

    def on_candle(self, candle: Candle):
        """Обновить состояние биржи и обработать Stop Loss/Take Profit.
        """
        self._current_candle = candle

        if not self.active_position:
            return

        # Иногда может происходить коллизия SL и TP (свеча достигла оба уровня и мы не знаем, что произошло первым)
        # В этом случае, намеренно обрабатываем худший случай (срабатываение Stop Loss первым), т.к.
        # лучше получить более пессимистичный результат тестирования, нежели более оптимистичный
        if self.active_position.stop_loss and candle.low <= self.active_position.stop_loss <= candle.high:

            self._close_position(price=self.active_position.stop_loss)

        elif self.active_position.take_profit and candle.low <= self.active_position.take_profit <= candle.high:

            self._close_position(price=self.active_position.take_profit)


class Strategy(ABC):

    def __init__(self):
        """Инициализация пользовательских параметров стратегии.
        Основная точка входа для пользователя системы. Здесь можно реализовать конфигурирование
        Stop Loss/Take Profit для позиций, параметры индикаторов и пр.
        (всего что относится непосредственно к торговле).
        """
        self.exchange: tp.Optional[Exchange] = None

    def setup(self, exchange: Exchange):
        """Инициализация окружения, в котором работает стратегия, подготовка стратегии к работе.
        Данный метод преимущественно должен вызываться на уровне системы (например, во время выполнения backtest-а).
        """
        self.exchange = exchange

        # TODO: Прогрев индикаторов (пока можно делать на уровне дочернего класса стратегии)

    @abstractmethod
    def on_candle(self, candle: Candle) -> None: ...

    async def backtest(
        self,
        ticker: str,
        timeframe: Timeframe,
        start_at: dt.datetime,
        end_at: dt.datetime
    ) -> 'AccountReport':
        return await backtest(self, ticker, timeframe, start_at, end_at)


@dataclass()
class AccountReport:
    _positions: tp.List[Position]
    _start_at: dt.datetime

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (profit=...)'

    @cached_property
    def equities(self):
        _equities = pd.Series(dtype=object)
        current_eq = EQUITY_INITIAL_AMOUNT

        initial_time = self._start_at.astimezone(settings.TIMEZONE)
        _equities[initial_time] = current_eq

        for position in self._positions:
            current_eq *= position.profit_ratio
            _equities[position.exit_time] = round(current_eq, 1)

        return _equities

    @cached_property
    def equity_graph(self) -> go.Figure:
        return graphs.get_report_graph(
            go.Scatter(x=self.equities.index, y=self.equities.values, name='Equity', line=dict(color='#7658e0'))
        )

    @cached_property
    def summary(self):
        df = pd.DataFrame.from_dict(p.as_dict() for p in self._positions)
        return {
            'mean_profit_ratio': RoundedDecimal(df.profit_ratio.mean()),
            'gmean_profit_ratio': RoundedDecimal(geometric_mean(df.profit_ratio)),
            'max_dropdown': df.profit_ratio.min(),
            'total_trades': int(df.profit_ratio.count()),
            'success_trades': int(df.profit_ratio[df.profit_ratio > 1].count()),
            'fail_trades': int(df.profit_ratio[df.profit_ratio <= 1].count()),
            'success_trades_ratio': RoundedDecimal(
                df.profit_ratio[df.profit_ratio > 1].count() / df.profit_ratio.count()
            ),
        }


async def backtest(
    strategy: Strategy,
    ticker: str,
    timeframe: Timeframe,
    start_at: dt.datetime,
    end_at: dt.datetime,
) -> AccountReport:

    exchange = Exchange()
    strategy.setup(exchange)

    async with BinanceClient() as api_client:
        logger.info('Fetching candles data...')
        candles = [
            candle async for candle in tqdm(api_client.get_futures_historical_candles(
                ticker=ticker, timeframe=timeframe, start=start_at, end=end_at
            ))
        ]

    logger.info('Perform strategy...')
    for candle in tqdm(candles):
        exchange.on_candle(candle)
        strategy.on_candle(candle)

    logger.info('Done!')
    return AccountReport(_positions=exchange.positions, _start_at=start_at)
