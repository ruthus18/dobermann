import asyncio
import datetime as dt
import logging.config
import typing as tp
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress

import simplejson as json
import zmq
from tortoise.queryset import QuerySet
from zmq.asyncio import Context

from app import db, models
from app.config import settings

from .base import Strategy, Ticker, Timeframe
from .utils import cancel_task, split_list_round_robin

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')


WORKERS_POOL_SIZE = 12

MANAGER_URL = 'tcp://127.0.0.1:4503'
FEED_URL = 'tcp://127.0.0.1:4504'


class Worker:

    def __init__(self, w_id: int, tickers: tp.List[Ticker]):
        self.w_id = w_id
        self.tickers = tickers

        self.log = logger.getChild(f'worker[{w_id}]')

    async def run(self):
        self.log.debug('Worker started')

        _zmq_ctx = Context.instance()
        sock = _zmq_ctx.socket(zmq.SUB)
        sock.connect(FEED_URL)
        sock.connect(MANAGER_URL)

        for ticker in self.tickers:
            sock.setsockopt_string(zmq.SUBSCRIBE, ticker)
            self.log.debug('Subscribed to candle %s', ticker)

        sock.setsockopt_string(zmq.SUBSCRIBE, '0')  # shutdown event

        self.log.debug('Waiting events...')
        while True:
            msg = await sock.recv_string()
            # self.log.debug('Got event: %s', msg)

            if msg == '0':
                sock.close()
                self.log.debug('Worker stopped')
                return

            ticker, candle = msg.split(';')


def spawn_worker(w_id: int, tickers: list):
    worker = Worker(w_id, tickers)
    try:
        asyncio.run(worker.run())
    except (KeyboardInterrupt, SystemExit):
        worker.log.debug('Force shutdown worker...')


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
                    pool, spawn_worker, w_id, tickers_splitted[w_id]
                )
                for w_id in range(pool_size)
            ]

            await asyncio.sleep(float('inf'))  # wait cancellation

        self.log.debug('Shutdown workers...')
        await manager_sock.send_string('0')

        await asyncio.wait(tasks)
        pool.shutdown(wait=True)

        self.log.debug('Manager stopped')


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
            strategy=None,
            start_at=dt.datetime(2021, 9, 1, tzinfo=settings.TIMEZONE),
            end_at=dt.datetime(2022, 2, 13, tzinfo=settings.TIMEZONE),
            timeframe=Timeframe.H1,
            tickers=['BTCUSDT', 'ETHUSDT', 'DYDXUSDT'],
        )
    )
