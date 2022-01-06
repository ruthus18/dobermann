import datetime as dt
import enum
import logging
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import cached_property

from pydantic import BaseModel
from tqdm.asyncio import tqdm

from .binance_client import BinanceClient, Candle, Timeframe
from .utils import OptDecimal, RoundedDecimal

logger = logging.getLogger(__name__)

BINANCE_FUTURES_LIMIT_COMISSION = Decimal('0.0002')
BINANCE_FUTURES_MARKET_COMISSION = Decimal('0.0004')


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


class Exchange:

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

    def open_market_position(
        self,
        position_type: PositionType = PositionType.LONG,
        stop_loss: OptDecimal = None,
        take_profit: OptDecimal = None,
    ) -> None:
        # Аналогично mk1
        ...

    def close_market_position(self) -> None:
        # Аналогично mk1
        ...

    def on_candle(self, candle: Candle) -> None:
        # Отслеживаем SL/TP текущей позиции
        ...

    def calculate_statistics(self) -> 'AccountReport':
        # TODO
        return AccountReport(_positions=self.positions)


class AccountReport(BaseModel):
    _positions: tp.List[Position]
    ...


class Strategy(ABC):

    def __init__(self):
        self.exchange: tp.Optional[Exchange] = None

        # Здесь инициализация пользовательских параметров для индикаторов и размер SL/TP

    def setup(self, exchange: Exchange):
        # Здесь инициализация окружения, в котором работает стратегия, а также прогрев индикаторов
        self.exchange = exchange

    @abstractmethod
    def on_candle(self, candle: Candle) -> None: ...


async def backtest(
    strategy: Strategy,
    symbol: str,
    timeframe: Timeframe,
    start_at: dt.datetime,
    end_at: dt.datetime,
) -> AccountReport:

    exchange = Exchange()
    strategy.setup(exchange)

    async with BinanceClient() as api_client:
        logger.info('Fetching candles data...')
        candles = [
            candle async for candle in api_client.get_futures_historical_candles(
                symbol=symbol, timeframe=timeframe, start=start_at, end=end_at
            )
        ]

    logger.info('Perform strategy...')
    for candle in tqdm(candles):
        exchange.on_candle(candle)
        strategy.on_candle(candle)

    return exchange.calculate_statistics()
