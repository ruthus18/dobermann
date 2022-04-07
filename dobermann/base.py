import datetime as dt
import importlib
import logging
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import cached_property
# from joblib import Parallel, delayed
import time
from statistics import geometric_mean

import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from app.config import settings  # FIXME

from . import graphs
from .binance_client import BinanceClient, Candle, Timeframe
from .utils import OptDecimal, RoundedDecimal, StrEnum

logger = logging.getLogger(__name__)


# TODO: configuration
BINANCE_FUTURES_LIMIT_COMISSION = Decimal('0.0002')
BINANCE_FUTURES_MARKET_COMISSION = Decimal('0.0004')

EQUITY_INITIAL_AMOUNT = Decimal(1000)


Ticker = str


class PositionType(StrEnum):
    LONG = 'LONG'
    SHORT = 'SHORT'


@dataclass
class Position:
    ticker: str
    type: PositionType

    enter_price: Decimal
    enter_time: dt.datetime
    enter_comission: Decimal

    exit_price: Decimal = None
    exit_time: dt.datetime = None
    exit_comission: Decimal = None

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
    """Интерфейс для стратегий, имитирующий работу с биржей.

    Предоставляет набор операций для открытия/закрытия позиций, отслеживания Stop Loss/Take Profit
    для открытых позиций и сбора статистики по торговому счету.

    ВАЖНО: Вызов метода `Exchange.on_candle()` должен происходить перед вызовом метода `Strategy.on_candle()`,
    т.к. для совершения действий на бирже сперва необходимо обновить состояние биржи (обработать текущие SL/TP и пр).
    """
    def __init__(
        self,
        market_order_comission: OptDecimal = BINANCE_FUTURES_MARKET_COMISSION,
        limit_order_comission: OptDecimal = BINANCE_FUTURES_LIMIT_COMISSION,
    ):
        self.market_order_comission = market_order_comission
        self.limit_order_comission = limit_order_comission

        self.positions: tp.List[Position] = []

        self.prices_now: tp.Dict[Ticker, Decimal] = {}
        self.time_now: tp.Optional[dt.datetime] = None

    # TODO: Объединить в open_position, limit или market - определяется на уровне реализации и передаваемых параметров 
    def open_market_position(
        self,
        ticker: str,
        position_type: PositionType = PositionType.LONG,
        *,
        stop_loss: OptDecimal = None,
        take_profit: OptDecimal = None,
    ) -> int:
        price_now = self.prices_now.get(ticker)

        if not price_now or not self.time_now:
            raise RuntimeError('Wrong state of exchange, you should call first `.on_candle()` method!')

        self.positions.append(
            Position(
                ticker=ticker,
                type=position_type,
                enter_price=price_now,
                enter_time=self.time_now,
                enter_comission=self.market_order_comission,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        )
        return len(self.positions) - 1  # position_id

    def close_market_position(self, position_id: int) -> None:
        # Стоимость = цене закрытия предыдущей свечи
        # (в данном случае делаем допущение, что время на получение свечи через API и генерацию сигнала = 0)
        try:
            position = self.positions[position_id]
        except IndexError:
            raise RuntimeError("Position wasn't open")

        if position.exit_time is not None:
            raise RuntimeError('Position already closed')

        price_now = self.prices_now.get(position.ticker)

        if not price_now or not self.time_now:
            raise RuntimeError('Wrong state of exchange, you should call first `.on_candle()` method!')

        position.exit_price = price_now
        position.exit_time = self.time_now
        position.exit_comission = self.market_order_comission

    def on_candle(self, candle: Candle, ticker: str) -> None:
        """Обновить состояние биржи.

        В отличие от `Strategy.on_candle`, в данный метод не передается `timeframe`, т.к. симуляция биржи
        работает с минимально допустимым таймфреймом.
        """
        self.prices_now[ticker] = candle.close

        open_time = candle.open_time

        if not self.time_now or open_time > self.time_now:
            self.time_now = open_time

        elif open_time < self.time_now:
            raise RuntimeError('Unable to turn back the time! Check your candles feeding')

    @property
    def closed_positions(self) -> tp.List[Position]:
        return [pos for pos in self.positions if pos.exit_time is not None]


class Strategy(ABC):

    def __init__(self):
        """Инициализация пользовательских параметров стратегии.

        Основная точка входа для пользователя системы. Здесь можно реализовать конфигурирование
        Stop Loss/Take Profit для позиций, параметры индикаторов и пр.
        (всего что относится непосредственно к торговле).
        """
        # self.exchange: tp.Optional[Exchange] = None
        # self.ticker: tp.Optional[str] = None

        # self._position_id: tp.Optional[int] = None

    # def setup(self, exchange: Exchange, ticker: str) -> None:
    #     """Инициализация окружения, в котором работает стратегия, подготовка стратегии к работе.

    #     Данный метод преимущественно должен вызываться на уровне системы (например, во время выполнения backtest-а).
    #     """
    #     self.exchange = exchange
    #     self.ticker = ticker
    #     # self.timeframe = timeframe

    #     # TODO: Прогрев индикаторов (пока можно делать на уровне дочернего класса стратегии)

    # async def backtest(
    #     self,
    #     ticker: str,
    #     timeframes: tp.List[Timeframe],
    #     start_at: dt.datetime,
    #     end_at: dt.datetime,
    # ) -> 'AccountReport':
    #     return await backtest(self, ticker, timeframes, start_at, end_at)

    # @property
    # def active_position(self) -> tp.Optional[Position]:
    #     if not self._position_id:
    #         return None

    #     return self.exchange.positions[self._position_id]

    # def open_market_position(
    #     self,
    #     position_type: PositionType = PositionType.LONG,
    #     stop_loss: OptDecimal = None,
    #     take_profit: OptDecimal = None,
    # ) -> None:
    #     self._position_id = self.exchange.open_market_position(
    #         self.ticker,
    #         position_type,
    #         stop_loss=stop_loss,
    #         take_profit=take_profit,
    #     )

    # def close_market_position(self) -> None:
    #     self.exchange.close_market_position(self._position_id)
    #     self._position_id = None

    @abstractmethod
    def on_candle(self, candle: Candle) -> None:
        """Точка входа для выполнения стратегии. Вызывается каждый раз при получии новой свечи.

        ВАЖНО:
        В самом простом случае, мы работаем на одном timeframe и аргумент `timeframe` можно не использовать
        (он всегда будет один и тот же). Если работа происходит на нескольких разных timeframe -- нужно обязательно
        это учитывать в логике стратегии.

        Также, при работе с разными timeframe не рекомендуется завязываться на порядок появления новых свечей. При
        реальной торговле нет абсолютно никаких гарантий, что например сперва будут поступать свечи 5m, а затем 1h.
        """
        raise NotImplementedError


@dataclass
class AccountReport:
    _positions: tp.List[Position]
    _start_at: dt.datetime

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (profit=...)'

    @cached_property
    def equities(self) -> pd.Series:
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
        return graphs.get_equity_graph(
            go.Scatter(x=self.equities.index, y=self.equities.values, name='Equity', line=dict(color='#7658e0'))
        )

    @cached_property
    def positions(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(p.as_dict() for p in self._positions)

    @cached_property
    def summary(self) -> dict:
        df = self.positions
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


# async def backtest(
#     strategy: Strategy,
#     assets: tp.List[Ticker],
#     timeframes: tp.List[Timeframe],
#     start_at: dt.datetime,
#     end_at: dt.datetime,
# ) -> AccountReport:
#     exchange = Exchange()
#     strategy.setup(exchange)

#     candles: tp.List[tp.Tuple[Candle, Timeframe]] = []

#     async with BinanceClient() as api_client:
#         for timeframe in timeframes:
#             logger.info('Fetching candles data for %s...', timeframe)
#             candles += [
#                 (candle, timeframe) async for candle in tqdm(api_client.get_futures_historical_candles(
#                     ticker=ticker, timeframe=timeframe, start=start_at, end=end_at
#                 ))
#             ]

#     candles.sort(key=lambda o: o[0].open_time)

#     logger.info('Perform strategy...')
#     for candle, timeframe in tqdm(candles):
#         if timeframe == timeframes[0]:  # TODO
#             exchange.on_candle(candle)

#         strategy.on_candle(candle, timeframe)

#     logger.info('Done!')
#     return AccountReport(_positions=exchange.closed_positions, _start_at=start_at)


POOL_SIZE = 6


async def backtest(
    strategy_cls: tp.Type[Strategy],
    tickers: tp.List[Ticker],
    timeframes: tp.List[Timeframe],
    start_at: dt.datetime,
    end_at: dt.datetime,
) -> AccountReport:
    started = time.time()

    candles = {}

    async with BinanceClient() as client:
        for ticker in tickers:
            for timeframe in timeframes:

                logger.info('Fetching candles data: a=%s, T=%s', ticker, timeframe)
                candles[(ticker, timeframe)] = [
                    candle async for candle in tqdm_async(client.get_futures_historical_candles(
                        ticker=ticker, timeframe=timeframe, start=start_at, end=end_at
                    ))
                ]

    pool = Parallel(n_jobs=POOL_SIZE)
    position_signals = pool(
        delayed(_backtest)(strategy_cls.__module__, strategy_cls.__name__, ticker, timeframe, candles)
    )

    elapsed = time.time() - started
    logger.info('Time elapsed: %.2fs', elapsed)

    return position_signals


# def _backtest(
#     strategy_path: str,
#     strategy_name: str,
#     ticker: Ticker,
#     timeframe: Timeframe,
#     candles: tp.List[Candle],
# ) -> pd.Series:
#     strategy_cls = getattr(importlib.import_module(strategy_path), strategy_name)
#     strategy = strategy_cls()

#     logger.info('Performing strategy: a=%s, T=%s', ticker, timeframe)
#     for candle in tqdm(candles):
#         strategy.on_candle(candle)

#     logger.info('Strategy finished: a=%s, T=%s', ticker, timeframe)
#     return strategy
