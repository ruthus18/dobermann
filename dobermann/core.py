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


"""
TODO:
* Механизм синхронизации между market feed и worker-ами (см. "slow joiner" синдром в ZMQ)
* Механизм завершения тестирования (доделать прототип через передачу последней даты свечи в worker-ы)
"""


WORKERS_POOL_SIZE = 12

_ADDR = 'tcp://127.0.0.1:{port}'

CANDLES_URL = _ADDR.format(port='4444')
FEED_REGISTRATOR_URL = _ADDR.format(port='4445')
WORKERS_CTRL_URL = _ADDR.format(port='4446')
EXCHANGE_URL = _ADDR.format(port='4447')


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
    ).encode()


def candle_from_json(data: bytes) -> Candle:
    return Candle(**json.loads(data.decode()))


class StrategyLogicalError(Exception): ...


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
        if self.position_id:
            raise StrategyLogicalError('Position already open')

        self.position_id = uuid.uuid4().hex

        await self.exchange_sock.send_json({
            'type': 'open',
            'position_id': self.position_id,
            'ticker': self.ticker,
            'time': self.candle.open_time,
            'price': self.candle.close,
        })

    async def close_position(self):
        if not self.position_id:
            raise StrategyLogicalError('Position was not open')

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
        ...  # TODO


class CandlesFeed:

    def __init__(
        self,
        start_at: dt.datetime,
        end_at: dt.datetime,
        timeframe: Timeframe,
        tickers: tp.List[Ticker],
        pool_size: int,
    ):
        super().__init__()

        self.start_at = start_at
        self.end_at = end_at
        self.timeframe = timeframe
        self.tickers: tp.Iterable[Ticker] = tickers  # FIXME: Сразу принимать на вход queryset с asset-ами
        self.pool_size = pool_size

        self.log = logger.getChild('feed')

        self.assets: QuerySet[models.Asset]
        self.candles_iter: tp.Awaitable[QuerySet[models.Candle]]

        self._zmq_ctx: zmq.Context
        self.candles_sender: zmq.Socket
        self.init_zmq()

    def init_zmq(self):
        self._zmq_ctx = Context.instance()

        self.candles_sender = self._zmq_ctx.socket(zmq.PUB)
        self.candles_sender.bind(CANDLES_URL)

    def close_zmq(self):
        self.candles_sender.close()

    async def _get_assets_from_db(self) -> QuerySet[models.Asset]:
        filter_kwargs = {
            'removed_at__isnull': True
        }
        if self.tickers is not None:
            filter_kwargs['ticker__in'] = self.tickers

        return await models.Asset.filter(**filter_kwargs)

    async def prepare_data(self):
        self.assets = await self._get_assets_from_db()
        self.log.debug('Assets loaded: %s', len(self.assets))

        self.candles_iter = models.Candle.filter(
            asset_id__in=[a.id for a in self.assets],
            timeframe=self.timeframe,
            open_time__gte=self.start_at,
            open_time__lt=self.end_at,
        ).select_related('asset').order_by('open_time')

        self.log.debug('Total candles: %s', await self.candles_iter.count())

    async def wait_actors(self):
        registrator = self._zmq_ctx.socket(zmq.REP)
        registrator.bind(FEED_REGISTRATOR_URL)

        workers_remaining = self.pool_size
        exchange_ready = False

        self.log.debug('Waiting workers and exchange to init...')

        while workers_remaining or not exchange_ready:
    
            actor_type, data = await registrator.recv_multipart()

            if actor_type == b'reg_worker':
                workers_remaining -= 1
                self.log.info('WORKER REGISTERED, REMAINING = %s', workers_remaining)

                await registrator.send_multipart((b'reg_worker', b'42'))

            elif actor_type == b'reg_exchange':
                self.log.info('EXCHANGE REGISTERED')
                exchange_ready = True

                await registrator.send_multipart((b'reg_exchange', b'3'))

        registrator.close()

    async def run(self):
        self.log.debug('Candles feed started')

        with suppress(asyncio.CancelledError):
            await self.prepare_data()
            await self.wait_actors()

            self.log.debug('Sending candles to workers...')
            async for candle in self.candles_iter:
                await self.candles_sender.send_multipart((candle.asset.ticker.encode(), candle_to_json(candle)))

            self.log.debug('Finish sending candles')
            await asyncio.sleep(float('inf'))  # wait cancellation

        self.close_zmq()
        self.log.debug('Candles feed stopped')


class Manager:

    def __init__(self, strategy: Strategy, tickers: tp.List[Ticker], pool_size: int):
        super().__init__()

        self.strategy = strategy
        self.tickers = tickers
        self.pool_size = pool_size

        self.log = logger.getChild('manager')

    async def run(self):
        self.log.debug('Manager started')

        _zmq_ctx = Context.instance()
        workers_sock = _zmq_ctx.socket(zmq.PUB)
        workers_sock.bind(WORKERS_CTRL_URL)

        tickers_splitted = split_list_round_robin(self.tickers, self.pool_size)

        loop = asyncio.get_event_loop()
        pool = ProcessPoolExecutor(max_workers=WORKERS_POOL_SIZE)

        with suppress(asyncio.CancelledError):
            tasks = [
                loop.run_in_executor(
                    pool, spawn_worker, w_id, self.strategy, tickers_splitted[w_id],
                )
                for w_id in range(self.pool_size)
            ]

            await asyncio.sleep(float('inf'))  # wait cancellation

        self.log.debug('Shutdown workers...')
        await workers_sock.send_string('exit')

        await asyncio.wait(tasks)
        pool.shutdown(wait=True)

        self.log.debug('Manager stopped')


def spawn_worker(w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
    worker = Worker(w_id, strategy, tickers)

    worker.log.debug('Worker started')
    try:
        asyncio.run(worker.run())

    except (KeyboardInterrupt, SystemExit):
        worker.log.debug('Force shutdown worker...')

    except Exception as e:
        worker.log.exception(
            'Exception occured while processing candles: (%s) %s', e.__class__.__name__, e
        )

    finally:
        worker.close_zmq()
        worker.log.debug('Worker stopped')


# TODO: Сохранение сигналов стратегий (передавать через сокет в менеджер, который будет писать в csv)
class Worker:

    def __init__(self, w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
        self.w_id = w_id
        self.tickers = tickers

        self.log = logger.getChild(f'worker[{w_id}]')

        self._strategy = strategy
        self.strategies: tp.Dict[Ticker, Strategy] = {}
        self.remaining_candles: tp.Dict[Ticker, Strategy] = {}

        self._zmq_ctx: zmq.Context
        self.poller: zmq.Poller
        self.registrator: zmq.Socket
        self.candle_receiver: zmq.Socket
        self.controller: zmq.Socket
        self.exchange_sender: zmq.Socket
        self.init_zmq()

    def init_zmq(self):
        self._zmq_ctx = Context.instance()

        self.candle_receiver = self._zmq_ctx.socket(zmq.SUB)
        self.candle_receiver.connect(CANDLES_URL)

        for ticker in self.tickers:
            self.candle_receiver.subscribe(ticker)
            self.log.debug('Subscribed to candle %s', ticker)

        self.controller = self._zmq_ctx.socket(zmq.SUB)
        self.controller.connect(WORKERS_CTRL_URL)
        self.controller.subscribe('')

        self.exchange_sender = self._zmq_ctx.socket(zmq.PUSH)
        self.exchange_sender.connect(EXCHANGE_URL)

        self.poller = zmq.Poller()
        self.poller.register(self.candle_receiver, zmq.POLLIN)
        self.poller.register(self.controller, zmq.POLLIN)

    def close_zmq(self):
        self.candle_receiver.close()
        self.controller.close()
        self.exchange_sender.close()
        self._zmq_ctx.term()

    async def register(self):
        registrator = self._zmq_ctx.socket(zmq.REQ)
        registrator.connect(FEED_REGISTRATOR_URL)

        await registrator.send_multipart((b'reg_worker', json.dumps(self.tickers).encode()))
        _, remaining = await registrator.recv_multipart()

        self.log.debug('Registered! Remaining candles: %s', remaining)

        registrator.close()

    async def run(self):
        for ticker in self.tickers:
            strategy = deepcopy(self._strategy)
            strategy._init_worker(ticker, self.exchange_sender)

            self.strategies[ticker] = strategy

        await self.register()

        self.log.debug('Waiting events...')
        while True:
            sockets = dict(self.poller.poll())

            if sockets.get(self.candle_receiver) == zmq.POLLIN:
                ticker, candle_data = await self.candle_receiver.recv_multipart()

                ticker = ticker.decode()
                candle = candle_from_json(candle_data)

                self.log.debug('Received candle %s: %s', ticker, candle)

                # TODO: check integrity errors in time series
                strategy = self.strategies[ticker]
                await strategy.on_candle(candle)

            if sockets.get(self.controller) == zmq.POLLIN:
                await self.controller.recv_string()

                self.log.debug('Received exit event')
                break


class Exchange:

    def __init__(self):
        self.log = logger.getChild('exchange')
        self._zmq_ctx = Context.instance()

    async def register(self):
        registrator = self._zmq_ctx.socket(zmq.REQ)
        registrator.connect(FEED_REGISTRATOR_URL)        

        await registrator.send_multipart((b'reg_exchange', b''))
        _, remaining = await registrator.recv_multipart()

        self.log.info('REGISTERED! REMAINING TICKERS: %s', remaining)

        registrator.close()

    async def run(self):
        self.log.debug('Exchange started')

        exchange_sock = self._zmq_ctx.socket(zmq.PULL)
        exchange_sock.bind(EXCHANGE_URL)

        with suppress(asyncio.CancelledError):
            await self.register()

            self.log.debug('Waiting events...')
            while True:
                msg = await exchange_sock.recv_json()
                logger.info('Got event: (%s)%s', type(msg), msg)

                ...  # TODO

        exchange_sock.close()
        self.log.debug('Exchange stopped')


async def run_trading_system(
    candles_feed: CandlesFeed,
    manager: Manager,
    exchange: Exchange,
):
    logger.info('Trading system started')
    t_manager = asyncio.create_task(manager.run())
    t_feed = asyncio.create_task(candles_feed.run())
    t_exchange = asyncio.create_task(exchange.run())

    try:
        await asyncio.wait(
            (t_manager, t_feed, t_exchange),
            return_when=asyncio.FIRST_COMPLETED
        )
    except (KeyboardInterrupt, SystemExit):
        pass

    finally:
        logger.debug('Waiting tasks to complete...')

        await cancel_task(t_manager)
        await cancel_task(t_feed)
        await cancel_task(t_exchange)

        logger.info('Trading system is stopped, goodbye 👋')


async def backtest(
    strategy: Strategy,
    start_at: dt.datetime,
    end_at: dt.datetime,
    timeframe: Timeframe,
    tickers: tp.Optional[tp.List[Ticker]],
):
    await db.init()

    if not start_at.tzinfo:
        start_at = start_at.astimezone(settings.TIMEZONE)

    if not end_at.tzinfo:
        end_at = end_at.astimezone(settings.TIMEZONE)

    pool_size = min(WORKERS_POOL_SIZE, len(tickers))

    candles_feed = CandlesFeed(start_at, end_at, timeframe, tickers, pool_size)
    manager = Manager(strategy, tickers, pool_size)
    exchange = Exchange()
    await run_trading_system(candles_feed, manager, exchange)

    await db.close()


if __name__ == '__main__':
    asyncio.run(
        backtest(
            strategy=TestStrategy(),
            start_at=dt.datetime(2022, 2, 1),
            end_at=dt.datetime(2022, 2, 3),
            timeframe=Timeframe.H1,
            tickers=['BTCUSDT', 'ETHUSDT', 'DYDXUSDT', 'NEARUSDT'],
        )
    )
