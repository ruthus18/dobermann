import asyncio
import datetime as dt
import logging.config
import typing as tp
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
import simplejson as json
import zmq
from tortoise.queryset import QuerySet
from zmq.asyncio import Context

from app import db, models
from app.config import settings

from . import indicators
from .base import Ticker, Timeframe
from .utils import cancel_task, split_list_round_robin

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')


WORKERS_POOL_SIZE = 12

MANAGER_URL = 'tcp://127.0.0.1:4444'
FEED_URL = 'tcp://127.0.0.1:4445'
EXCHANGE_URL = 'tcp://127.0.0.1:4446'


SignalType = tp.TypeVar('SignalType')


@dataclass
class Candle:
    open_time: dt.datetime
    close_time: dt.datetime
    open: Decimal
    close: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal


def candle_to_json(obj: models.Candle) -> str:
    return json.dumps(
        dict(
            open_time=obj.open_time,
            close_time=obj.close_time,
            volume=obj.volume,
            open=obj.open,
            close=obj.close,
            low=obj.low,
            high=obj.high,
        ),
        default=str,
    )


def candle_from_json(data: str) -> Candle:
    return Candle(**json.loads(data))


class Strategy(ABC):

    def __init__(self):
        self.ticker: tp.Optional[Ticker] = None
        self.exchange_sock: tp.Optional[zmq.Socket] = None

        self.candle: tp.Optional[dict] = None
        self.position_id: tp.Optional[str] = None

        self.signals: tp.Dict[dt.datetime: SignalType] = {}
        

    def _init_worker(self, ticker: Ticker, exchange_sock: zmq.Socket):
        self.ticker = ticker
        self._exchange_sock = exchange_sock

    # TODO: Стратегия не должна хранить в себе логику работы с биржей, этим должен заниматься Worker (ExchangeClient)
    async def open_position(self):
        if self.position_id is not None:
            raise RuntimeError('Position already open')

        self.position_id = uuid.uuid4().hex

        await self.exchange_sock.send_json({
            'type': 'open',
            'position_id': self.position_id,
            'ticker': self.ticker,
            'time': self.candle.open_time,
            'price': self.candle.close,
        })

    async def close_position(self):
        if self.position_id is None:
            raise RuntimeError('Position not open')

        await self.exchange_sock.send_json({
            'type': 'close',
            'position_id': self.position_id,
            'time': self.candle.open_time,
            'price': self.candle.price,
        })
        self.position_id = None

    async def on_candle(self, candle: Candle):
        self.candle = candle
        signal = await self.calculate()
        self.signals[candle.close_time] = signal

    @abstractmethod
    async def calculate(self) -> SignalType: ...


class TestStrategy(Strategy):
    def __init__(self):
        super().__init__()

    async def calculate(self) -> SignalType:
        return await super().calculate()


class CandlesFeed:

    def __init__(
        self,
        start_at: dt.datetime,
        end_at: dt.datetime,
        timeframe: Timeframe,
        tickers: tp.Optional[tp.List[Ticker]] = None,
    ):
        super().__init__()

        self.start_at = start_at
        self.end_at = end_at

        self.tickers: tp.Iterable[Ticker] = tickers
        self.assets: QuerySet[models.Asset]

        self.timeframe = timeframe

        self.log = logger.getChild('feed')

    async def get_assets_from_db(self) -> QuerySet[models.Asset]:
        filter_kwargs = {
            'removed_at__isnull': True
        }
        if self.tickers is not None:
            filter_kwargs['ticker__in'] = self.tickers

        return await models.Asset.filter(**filter_kwargs)

    async def run(self):
        _zmq_ctx = Context.instance()
        feed_sock = _zmq_ctx.socket(zmq.PUB)
        feed_sock.bind(FEED_URL)

        self.log.debug('Candles feed started')

        with suppress(asyncio.CancelledError):

            self.assets = await self.get_assets_from_db()
            self.log.debug('Assets loaded: %s', len(self.assets))

            candles_iter = models.Candle.filter(
                asset__ticker__in=[a.ticker for a in self.assets],
                timeframe=self.timeframe,
                open_time__gte=self.start_at,
                open_time__lte=self.end_at,
            ).select_related('asset').order_by('open_time')


            self.log.debug('Sending candles to workers...')
            self.log.debug('Total candles: %s', await candles_iter.count())

            async for candle in candles_iter:
                json_msg = candle_to_json(candle)

                await feed_sock.send_string(f'{candle.asset.ticker};{json_msg}')

            self.log.debug('Finish sending candles')
            await asyncio.sleep(float('inf'))  # wait cancellation

        self.log.debug('Candles feed stopped')


class Manager:

    def __init__(self, strategy: Strategy, tickers: tp.List[Ticker]):
        super().__init__()

        self.strategy = strategy
        self.tickers = tickers

        self.log = logger.getChild('manager')

    async def run(self):
        self.log.debug('Manager started')

        _zmq_ctx = Context.instance()
        manager_sock = _zmq_ctx.socket(zmq.PUB)
        manager_sock.bind(MANAGER_URL)

        pool_size = min(WORKERS_POOL_SIZE, len(self.tickers))
        tickers_splitted = split_list_round_robin(self.tickers, pool_size)

        loop = asyncio.get_event_loop()
        pool = ProcessPoolExecutor(max_workers=WORKERS_POOL_SIZE)

        with suppress(asyncio.CancelledError):
            tasks = [
                loop.run_in_executor(
                    pool, spawn_worker, w_id, self.strategy, tickers_splitted[w_id],
                )
                for w_id in range(pool_size)
            ]

            await asyncio.sleep(float('inf'))  # wait cancellation

        self.log.debug('Shutdown workers...')
        await manager_sock.send_string('0')

        await asyncio.wait(tasks)
        pool.shutdown(wait=True)

        self.log.debug('Manager stopped')


def spawn_worker(w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
    worker = Worker(w_id, strategy, tickers)
    try:
        asyncio.run(worker.run())
    except (KeyboardInterrupt, SystemExit):
        worker.log.debug('Force shutdown worker...')


class Worker:

    def __init__(self, w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
        self.w_id = w_id
        self.tickers = tickers

        self.log = logger.getChild(f'worker[{w_id}]')

        self._strategy = strategy
        self.strategies: tp.Dict[Ticker, Strategy] = {}

    async def run(self):
        self.log.debug('Worker started')

        _zmq_ctx = Context.instance()
        sock = _zmq_ctx.socket(zmq.SUB)
        sock.connect(FEED_URL)
        sock.connect(MANAGER_URL)

        exchange_sock = _zmq_ctx.socket(zmq.PUSH)
        exchange_sock.connect(EXCHANGE_URL)

        for ticker in self.tickers:
            sock.setsockopt_string(zmq.SUBSCRIBE, ticker)

            strategy = deepcopy(self._strategy)
            strategy._init_worker(ticker, exchange_sock)

            self.strategies[ticker] = strategy
            self.log.debug('Subscribed to candle %s', ticker)

        sock.setsockopt_string(zmq.SUBSCRIBE, '0')  # shutdown event

        self.log.debug('Waiting events...')
        while True:
            msg = await sock.recv_string()
            self.log.debug('Got event: %s', msg)

            if msg == '0':
                sock.close()
                self.log.debug('Worker stopped')
                break

            ticker, candle_data = msg.split(';')

            strategy = self.strategies[ticker]
            await strategy.on_candle(candle_from_json(candle_data))

        # TODO: Сохранение сигналов из стратегии в БД


async def run_trading_system(
    candles_feed: CandlesFeed,
    manager: Manager,
):
    logger.info('Trading system started')
    t_manager = asyncio.create_task(manager.run())
    t_other = asyncio.create_task(candles_feed.run())

    try:
        await asyncio.wait(
            [t_manager, t_other],
            return_when=asyncio.FIRST_COMPLETED
        )
    except (KeyboardInterrupt, SystemExit):
        pass

    finally:
        logger.debug('Waiting tasks to complete...')

        await cancel_task(t_manager)
        await cancel_task(t_other)

        logger.info('Trading system is stopped, goodbye 👋')


async def backtest(
    strategy: Strategy,
    start_at: dt.datetime,
    end_at: dt.datetime,
    timeframe: Timeframe,
    tickers: tp.Optional[tp.List[Ticker]] = None,
):
    await db.init()

    candles_feed = CandlesFeed(start_at, end_at, timeframe, tickers)
    manager = Manager(strategy, tickers)
    await run_trading_system(candles_feed, manager)

    await db.close()


if __name__ == '__main__':
    asyncio.run(
        backtest(
            strategy=TestStrategy(),
            start_at=dt.datetime(2021, 9, 1, tzinfo=settings.TIMEZONE),
            end_at=dt.datetime(2022, 2, 13, tzinfo=settings.TIMEZONE),
            timeframe=Timeframe.H1,
            tickers=['BTCUSDT', 'ETHUSDT', 'DYDXUSDT'],
        )
    )
