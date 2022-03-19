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

FEED_URL = 'tcp://127.0.0.1:4444'
WORKERS_CTRL_URL = 'tcp://127.0.0.1:4445'
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
        tickers: tp.Optional[tp.List[Ticker]] = None,
    ):
        super().__init__()

        self.start_at = start_at
        self.end_at = end_at
        self.timeframe = timeframe
        self.tickers: tp.Iterable[Ticker] = tickers or []  # FIXME: Сразу принимать на вход queryset с asset-ами

        self.assets: QuerySet[models.Asset]
        self.candles_iter: tp.Awaitable[QuerySet[models.Candle]]

        self.log = logger.getChild('feed')

    async def _get_assets_from_db(self) -> QuerySet[models.Asset]:
        filter_kwargs = {
            'removed_at__isnull': True
        }
        if self.tickers is not None:
            filter_kwargs['ticker__in'] = self.tickers

        return await models.Asset.filter(**filter_kwargs)

    async def _prepare_data(self):
        self.assets = await self._get_assets_from_db()
        self.log.debug('Assets loaded: %s', len(self.assets))

        self.candles_iter = models.Candle.filter(
            asset_id__in=[a.id for a in self.assets],
            timeframe=self.timeframe,
            open_time__gte=self.start_at,
            open_time__lt=self.end_at,
        ).select_related('asset').order_by('open_time')

        self.log.debug('Total candles: %s', await self.candles_iter.count())

    async def run(self):
        _zmq_ctx = Context.instance()
        feed_sock: zmq.Socket = _zmq_ctx.socket(zmq.PUB)
        feed_sock.bind(FEED_URL)

        self.log.debug('Candles feed started')

        with suppress(asyncio.CancelledError):
            await self._prepare_data()
            await asyncio.sleep(3)  # TODO: workers handshake

            self.log.debug('Sending candles to workers...')
            async for candle in self.candles_iter:
                await feed_sock.send_multipart((candle.asset.ticker.encode(), candle_to_json(candle)))

            self.log.debug('Finish sending candles')
            await asyncio.sleep(float('inf'))  # wait cancellation

        feed_sock.close()
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
        workers_sock = _zmq_ctx.socket(zmq.PUB)
        workers_sock.bind(WORKERS_CTRL_URL)

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


# FIXME: Нужно переделать формат входящих событий и вынести сообщения в отдельный объект (CandleEvent):
#        <ticker>;<action>;<data>
#        ------------------------
#        BTCUSDT;last_open_at;2022-03-10T06:20:56.915248
#        ETHUSDT;candle;{...}
#
# TODO: Сохранение сигналов стратегий (передавать через сокет в менеджер, который будет писать в csv)
class Worker:

    def __init__(self, w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
        self.w_id = w_id
        self.tickers = tickers

        self.log = logger.getChild(f'worker[{w_id}]')

        self._strategy = strategy
        self.strategies: tp.Dict[Ticker, Strategy] = {}

        self._zmq_ctx: zmq.Context
        self.poller: zmq.Poller
        self.candle_receiver: zmq.Socket
        self.controller: zmq.Socket
        self.exchange_sender: zmq.Socket
        self.init_zmq()

    def init_zmq(self):
        self._zmq_ctx = Context.instance()

        self.candle_receiver = self._zmq_ctx.socket(zmq.SUB)
        self.candle_receiver.connect(FEED_URL)

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

    async def run(self):
        for ticker in self.tickers:
            strategy = deepcopy(self._strategy)
            strategy._init_worker(ticker, self.exchange_sender)

            self.strategies[ticker] = strategy

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

    async def run(self):
        self.log.debug('Exchange started')

        _zmq_ctx = Context.instance()
        exchange_sock = _zmq_ctx.socket(zmq.PULL)
        exchange_sock.bind(EXCHANGE_URL)

        with suppress(asyncio.CancelledError):
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
    tickers: tp.Optional[tp.List[Ticker]] = None,
):
    await db.init()

    if not start_at.tzinfo:
        start_at = start_at.astimezone(settings.TIMEZONE)

    if not end_at.tzinfo:
        end_at = end_at.astimezone(settings.TIMEZONE)

    candles_feed = CandlesFeed(start_at, end_at, timeframe, tickers)
    manager = Manager(strategy, tickers)
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
