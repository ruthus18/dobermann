import asyncio
import datetime as dt
import logging.config
import signal
import time
import typing as tp
from abc import ABC, abstractmethod
from copy import copy
import multiprocessing as mp
import multiprocessing.connection as mp_connection

import simplejson as json
import zmq
from tortoise.queryset import QuerySet
from zmq.asyncio import Context

from app import db, models
from app.config import settings

from .base import Strategy, Ticker, Timeframe
from .utils import cancel_task

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')


LOGS_URL = 'tcp://127.0.0.1:7267'
FEED_URL = 'tcp://127.0.0.1:7268'
WORKER_URL = 'tcp://127.0.0.1:7269'

WORKER_PROCESSES_NUM = 10


class Actor(ABC):
    name = NotImplemented

    def __init__(self):
        self.zmq_ctx = Context.instance()
        self._logs_sock = self.zmq_ctx.socket(zmq.PUSH)

    async def log(self, message: str):
        await self._logs_sock.send_string(f'{self.name};{message}')

    async def setup(self):
        self._logs_sock.connect(LOGS_URL)
        await self.log('Connected')

    async def shutdown(self):
        await self.log('Disconnected')
        await asyncio.sleep(0.01)  # wait while log sending through socket
        self._logs_sock.close()

    @abstractmethod
    async def run(self): ...


class MarketFeed(Actor):
    """Base class for supplying market data to trading system.
    """
    name = 'feed'


def candle_to_json(obj: models.Candle) -> str:
    return json.dumps(
        dict(
            ticker=obj.asset.ticker,
            timeframe=obj.timeframe,
            open_time=obj.open_time,
            open=obj.open,
            close=obj.close,
            low=obj.low,
            high=obj.high,
        ),
        default=str,
    )


class BacktestingMarketFeed(MarketFeed):

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

        self.feed_sock = self.zmq_ctx.socket(zmq.PUSH)

    async def setup(self):
        self.feed_sock.bind(FEED_URL)
        await super().setup()

    async def shutdown(self):
        self.feed_sock.close()
        await super().shutdown()

    async def get_assets_from_db(self) -> QuerySet[models.Asset]:
        filter_kwargs = {
            'removed_at__isnull': True
        }
        if self.tickers is not None:
            filter_kwargs['ticker__in'] = self.tickers

        return await models.Asset.filter(**filter_kwargs)

    async def run(self):
        self.assets = await self.get_assets_from_db()
        await self.log(f'Assets loaded: {len(self.assets)}')

        candles_iter = models.Candle.filter(
            asset__ticker__in=[a.ticker for a in self.assets],
            timeframe=self.timeframe,
            open_time__gte=self.start_at,
            open_time__lte=self.end_at,
        ).select_related('asset').order_by('open_time')

        async for candle in candles_iter:
            json_msg = candle_to_json(candle)

            await self.feed_sock.send_string(f'{candle.asset.ticker};{json_msg}')

        await asyncio.sleep(float('inf'))  # wait cancellation


class StrategyDispatcher(Actor):
    name = 'dispatcher'

    def __init__(self, strategy: Strategy, tickers: tp.List[Ticker], workers_n: int):
        super().__init__()

        self.strategy = strategy
        self.tickers = tickers
        self.workers_n = workers_n

        self.feed_sock = self.zmq_ctx.socket(zmq.PULL)
        self.worker_sock = self.zmq_ctx.socket(zmq.PUB)

        self.router = {}  # ticker -> worker_id
        self.workers: tp.List[mp.Process] = []

    async def setup_workers(self):
        await self.log('Launching workers...')

        workers_num = min(self.workers_n, len(self.tickers))

        for worker_id in range(workers_num):
            worker = mp.Process(target=StrategyWorker.spawn, args=(copy(self.strategy), worker_id))
            worker.start()

            self.workers.append(worker)

    async def setup(self):
        self.worker_sock.bind(WORKER_URL)
        self.feed_sock.connect(FEED_URL)
        await super().setup()

        await self.setup_workers()

    async def shutdown_workers(self):
        await self.log('Waiting workers to shutdown...')
        for worker in self.workers:
            worker.terminate()

        mp_connection.wait(worker.sentinel for worker in self.workers)

    async def shutdown(self):
        await self.shutdown_workers()

        self.feed_sock.close()
        self.worker_sock.close()

        await super().shutdown()

    async def run(self):
        await asyncio.sleep(5)

        while True:
            await asyncio.sleep(float('inf'))  # wait cancellation


class StrategyWorker(Actor):
    _name = 'worker[{id}]'
    
    def __init__(self, strategy: Strategy, w_id: int):
        super().__init__()

        self.strategy = strategy
        self.w_id = w_id

        self.worker_sock = self.zmq_ctx.socket(zmq.SUB)

    @property
    def name(self):
        return self._name.format(id=self.w_id)

    async def setup(self):
        self.worker_sock.connect(WORKER_URL)
        await super().setup()

    async def shutdown(self):
        self.worker_sock.close()
        await super().shutdown()

    async def _spawn(self):
        loop = asyncio.get_event_loop()

        stop_event = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            loop.add_signal_handler(sig, stop_event.set)

        await self.setup()

        main_task = asyncio.create_task(self.run())
        stop_event_task = asyncio.create_task(stop_event.wait())

        try:
            await asyncio.wait(
                (main_task, stop_event_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            async with cancel_task(main_task):
                await self.shutdown()

    @classmethod
    def spawn(cls, *args):
        self = cls(*args)
        asyncio.run(self._spawn())

    async def run(self):
        while True:
            msg = await self.worker_sock.recv_string()
            await self.log(f'Got message: {msg}')


async def logs_watcher(logs_sock: zmq.Socket) -> None:
    while True:
        event = await logs_sock.recv_string()
        name, msg = event.split(';')

        logger.getChild(name).info(msg)


async def run_trading_system(
    market_feed: MarketFeed,
    dispatcher: StrategyDispatcher,
):
    logger.info('Starting trading system')

    loop = asyncio.get_event_loop()

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, stop_event.set)

    _zmq_ctx = Context.instance()
    logs_sock = _zmq_ctx.socket(zmq.PULL)
    logs_sock.bind(LOGS_URL)

    await market_feed.setup()
    await dispatcher.setup()

    logger_task = asyncio.create_task(logs_watcher(logs_sock))

    feed_task = asyncio.create_task(market_feed.run())
    dispatcher_task = asyncio.create_task(dispatcher.run())

    wait_stop_task = asyncio.create_task(stop_event.wait())

    try:
        await asyncio.wait(
            (wait_stop_task, feed_task, dispatcher_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        logger.info('Shutdown system...')

    finally:
        async with cancel_task(dispatcher_task):
            await dispatcher.shutdown()

        async with cancel_task(feed_task):
            await market_feed.shutdown()

        async with cancel_task(logger_task):
            logs_sock.close()

    logger.info('Trading system is stopped')


async def backtest(
    strategy: Strategy,
    start_at: dt.datetime,
    end_at: dt.datetime,
    timeframe: Timeframe,
    tickers: tp.Optional[tp.List[Ticker]] = None,  # if None -> load every available asset from DB
):
    market_feed = BacktestingMarketFeed(
        start_at=start_at,
        end_at=end_at,
        timeframe=timeframe,
        tickers=tickers,
    )
    dispatcher = StrategyDispatcher(
        strategy=strategy,
        tickers=tickers,
        workers_n=WORKER_PROCESSES_NUM,
    )

    logger.info('Backtesting started')

    elapsed = time.time()
    await run_trading_system(
        market_feed=market_feed,
        dispatcher=dispatcher,
    )
    logger.info('Backtesting complete in %.2fs.', time.time() - elapsed)


async def main():
    await db.init()
    await backtest(
        strategy=None,
        start_at=dt.datetime(2021, 9, 1, tzinfo=settings.TIMEZONE),
        end_at=dt.datetime(2022, 2, 13, tzinfo=settings.TIMEZONE),
        timeframe=Timeframe.H1,
        tickers=['BTCUSDT', 'ETHUSDT', 'DYDXUSDT'],
    )
    await db.close()


if __name__ == '__main__':
    asyncio.run(main())
