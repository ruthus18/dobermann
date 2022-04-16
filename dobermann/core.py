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

import pytz
import simplejson as json
import zmq
from zmq.asyncio import Context, Poller

from . import db, utils
from .binance_client import Timeframe
from .config import settings

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')  # TODO: Затянуть либу loguru


Ticker = str
AssetID = int
SignalType = tp.TypeVar('SignalType')


_ZMQ_URL = 'tcp://127.0.0.1:{port}'

CANDLES_URL = _ZMQ_URL.format(port=22222)
EXCHANGE_URL = _ZMQ_URL.format(port=22223)
WORKERS_CONTROLLER_URL = _ZMQ_URL.format(port=22224)
WORKERS_RESPONDER_URL = _ZMQ_URL.format(port=22225)

MAX_WORKERS_POOL_SIZE = 12

EVENT_ALL_CANDLES_SENT = b'-1'

TZ = pytz.timezone(settings.TZ_NAME)


@dataclass(slots=True)
class Candle:
    timeframe: Timeframe
    open_time: dt.datetime
    open: Decimal
    close: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal

    @classmethod
    def from_dict(cls, d: dict) -> 'Candle':
        return cls(**{
            **d,
            'open_time': d['open_time'].astimezone(TZ)
        })


class StrategyRuntimeError(Exception): ...


class Strategy(ABC):

    def __init__(self):
        self.ticker: Ticker = None
        self.exchange: 'ExchangeClient' = None

        self.candle: Candle = None
        self.position_id: str = None

    def _init_worker(self, ticker: Ticker, exchange_client: 'ExchangeClient'):
        self.ticker = ticker
        self.exchange = exchange_client

    # TODO: стратегия не должна хранить логику клиента, нужно просто прокисровать вызов в клиент
    async def open_position(self):
        if self.position_id:
            raise StrategyRuntimeError('Position already open')

        self.position_id = uuid.uuid4().hex

        await self.exchange.open_position({
            'type': 'open',
            'position_id': self.position_id,
            'ticker': self.ticker,
            'time': self.candle.open_time,
            'price': self.candle.close,
        })

    # TODO: стратегия не должна хранить логику клиента, нужно просто прокисровать вызов в клиент
    async def close_position(self):
        if not self.position_id:
            raise StrategyRuntimeError('Position was not open')

        await self.exchange.close_position({
            'type': 'close',
            'position_id': self.position_id,
            'time': self.candle.open_time,
            'price': self.candle.price,
        })
        self.position_id = None

    async def _on_candle(self, candle: Candle) -> SignalType:
        self.candle = candle
        return await self.on_candle(candle)

    @abstractmethod
    async def on_candle(self, candle: Candle) -> SignalType: ...


class TestStrategy(Strategy):

    async def on_candle(self, candle):
        return None


class CandlesFeed:

    def __init__(
        self,
        start_at: dt.datetime,
        end_at: dt.datetime,
        timeframe: Timeframe,
        tickers: list[Ticker] | None,
    ):
        if start_at.tzinfo:
            self.start_at = start_at
        else:
            self.start_at = start_at.astimezone(settings.TIMEZONE)

        if end_at.tzinfo:
            self.end_at = end_at
        else:
            self.end_at = end_at.astimezone(settings.TIMEZONE)

        self.timeframe = timeframe
        self.tickers = tickers
        self._assets: dict[Ticker, AssetID] = None

        self.log = logger.getChild('feed')

        self.zmq_ctx = Context.instance()

        self.candles_sender = self.zmq_ctx.socket(zmq.PUB)
        self.candles_sender.bind(CANDLES_URL)
        self.candles_sender.set(zmq.SNDHWM, 0)

        self._total_candles_sent = 0

    async def get_assets(self) -> dict[AssetID, Ticker]:
        if self._assets:
            return self._assets

        query = f'''
        SELECT asset.id, asset.ticker FROM asset
            LEFT JOIN candle
                ON candle.asset_id = asset.id
                AND candle.timeframe = $1
                AND candle.open_time >= $2
                AND candle.open_time < $3
                {'AND asset.ticker = ANY($4::text[])' if self.tickers else ''}
        GROUP BY asset.id
        HAVING COUNT(candle.id) > 0;
        '''
        args = [self.timeframe, self.start_at, self.end_at]
        if self.tickers:
            args.append(self.tickers)

        self.log.debug('Loading assets...')
        async with db.cursor(query, *args) as cursor:
            assets = await cursor.fetch(10 ** 4)

        self._assets = dict(assets)
        self.log.debug('Assets loaded: total=%s', len(assets))

        return self._assets

    async def get_actual_tickers(self) -> set[Ticker]:
        assets = await self.get_assets()
        return set(assets.values())

    async def send_candles(self):
        assets = await self.get_assets()
        query = '''
        SELECT asset_id, timeframe, open_time, open, close, low, high, volume FROM candle
            WHERE candle.asset_id = ANY($1::int[])
                AND candle.timeframe = $2
                AND candle.open_time >= $3
                AND candle.open_time < $4
        ORDER BY candle.open_time ASC;
        '''
        args = (assets.keys(), self.timeframe, self.start_at, self.end_at)

        async with db.connection() as conn:
            # FIXME: do we really need this premature optimization??? (test)
            await utils.disable_decimal_conversion_codec(conn)

            async with conn.transaction():
                self.log.debug('Preparing candles to send...')
                cursor = await conn.cursor(query, *args)

                self.log.debug('Sending candles...')
                while candles_batch := await cursor.fetch(1000):
                    for candle in candles_batch:

                        ticker = assets[candle['asset_id']]

                        await self.candles_sender.send_multipart((
                            ticker.encode(), utils.packb_candle(candle)
                        ))

                        self._total_candles_sent += 1

        await self.candles_sender.send(EVENT_ALL_CANDLES_SENT)
        self.log.debug('All candles sent: total=%s', self._total_candles_sent)

    async def run(self):
        self.log.debug('Candles feed started')

        with suppress(asyncio.CancelledError):
            await self.register_candle_recipients()
            await self.send_candles()

        self.log.debug('Candles feed stopped')

    def close(self):
        self.candles_sender.close()


class WorkManager:

    def __init__(self, tickers: list[Ticker], strategy: Strategy, *, pool_size: int | None = None):
        self.tickers = tickers
        self.strategy = strategy
        self.pool_size = pool_size or min(len(tickers), MAX_WORKERS_POOL_SIZE)

        self.workers: list[asyncio.Future] = None
        self.proc_executor = ProcessPoolExecutor(max_workers=pool_size)

        self.log = logger.getChild('manager')

        self.zmq_ctx = Context.instance()

        self.workers_controller = self.zmq_ctx.socket(zmq.PUB)
        self.workers_controller.bind(WORKERS_CONTROLLER_URL)

        self.workers_responder = self.zmq_ctx.socket(zmq.PULL)
        self.workers_responder.bind(WORKERS_RESPONDER_URL)

    async def start_workers(self):
        if self.workers:
            raise RuntimeError('Workers already running')

        loop = asyncio.get_event_loop()
        tickers_splitted = utils.split_list_round_robin(self.tickers, self.pool_size)

        self.log.debug('Starting workers...')
        self.workers = [
            loop.run_in_executor(
                self.proc_executor, spawn_worker, w_id, tickers_splitted[w_id], self.strategy
            )
            for w_id in range(self.pool_size)
        ]
        for _ in range(self.pool_size):
            await self.workers_responder.recv()

        self.log.debug('Workers started')

    async def wait_workers(self):
        if not self.workers:
            raise RuntimeError('Workers was not started')

        self.log.debug('Waiting workers to complete...')

        await asyncio.wait(self.workers)
        self.log.debug('Workers compelted')

    async def shutdown_workers(self):
        if not self.workers:
            raise RuntimeError('Workers was not started')

        if not all(worker.done() for worker in self.workers):
            self.log.debug('Performing force shutdown of workers...')

            await self.workers_controller.send_string('exit')
            await asyncio.wait(self.workers)

        self.proc_executor.shutdown(wait=True)
        self.log.debug('Workers shutdown')

    def close(self):
        self.workers_controller.close()
        self.workers_responder.close()


def spawn_worker(w_id: int, tickers: list[Ticker], strategy: Strategy):
    worker = Worker(w_id, tickers, strategy)
    worker.log.debug('Worker started')

    try:
        asyncio.run(worker.listen_events())

    except (KeyboardInterrupt, SystemExit):
        worker.log.debug('Force shutdown worker...')

    except Exception as e:
        worker.log.exception(
            'Exception occured while processing candles: (%s) %s', e.__class__.__name__, e
        )

    finally:
        worker.close()
        worker.log.debug('Worker stopped')


class Worker:

    def __init__(self, w_id: int, tickers: list[Ticker],  strategy: Strategy):
        self.w_id = w_id
        self.tickers = tickers

        self._strategy = strategy
        self.strategies: dict[Ticker, Strategy] = {}

        self._total_candles_processed = 0

        self.log = logger.getChild(f'worker[{w_id}]')

        self.zmq_ctx = Context.instance()

        self.exchange_client = ExchangeClient(zmq_ctx=self.zmq_ctx)

        self.candles_receiver = self.zmq_ctx.socket(zmq.SUB)
        self.candles_receiver.connect(CANDLES_URL)
        self.candles_receiver.set(zmq.RCVHWM, 0)

        self.subscribe_to_candles()
        self.candles_receiver.subscribe(EVENT_ALL_CANDLES_SENT)

        self.controller = self.zmq_ctx.socket(zmq.SUB)
        self.controller.connect(WORKERS_CONTROLLER_URL)
        self.controller.subscribe('')

        self.poller = Poller()
        self.poller.register(self.candles_receiver, zmq.POLLIN)
        self.poller.register(self.controller, zmq.POLLIN)

    def subscribe_to_candles(self):
        for ticker in self.tickers:
            self.candles_receiver.subscribe(ticker)

            strategy = deepcopy(self._strategy)
            strategy._init_worker(ticker, self.exchange_client)
            self.strategies[ticker] = strategy

    async def notify_ready(self):
        responder = self.zmq_ctx.socket(zmq.PUSH)
        responder.connect(WORKERS_RESPONDER_URL)

        await responder.send(b'')
        responder.close()

    async def listen_events(self):
        await self.notify_ready()

        while sockets := dict(await self.poller.poll()):

            if sockets.get(self.candles_receiver) == zmq.POLLIN:
                msg = await self.candles_receiver.recv_multipart()
                match msg:
                    case [ticker, candle_data]:
                        self._total_candles_processed += 1

                        ticker = ticker.decode()
                        candle = Candle.from_dict(utils.unpackb_candle(candle_data))

                        await self.process_candle(ticker, candle)

                    case [EVENT_ALL_CANDLES_SENT]:
                        self.log.debug('All candles processed: total=%s', self._total_candles_processed)
                        return

            if sockets.get(self.controller) == zmq.POLLIN:
                await self.controller.recv_string()

                self.log.debug('Received exit event')
                return

    async def process_candle(self, ticker: str, candle: Candle):
        strategy = self.strategies[ticker]
        signal = await strategy._on_candle(candle)  # TODO: writing signals

    def close(self):
        self.candles_receiver.close()
        self.controller.close()


class ExchangeClient:

    def __init__(self, zmq_ctx: zmq.Context):
        self.zmq_ctx = zmq_ctx

        self.sender: zmq.Socket = self.zmq_ctx.socket(zmq.PUSH)
        self.sender.connect(EXCHANGE_URL)

    def close(self):
        self.sender.close()

    async def send_event(self, event_name: str, event_data: dict):
        await self.sender.send_multipart((event_name.encode(), json.dumps(event_data).encode()))

    async def open_position(self, data: dict):
        await self.send_event('open_position', data)

    async def close_position(self, data: dict):
        await self.send_event('close_position', data)

    async def finish_processing(self, ticker: str):
        await self.send_event('finish_processing', {'ticker': ticker})


async def backtest(
    strategy: Strategy,
    start_at: dt.datetime,
    end_at: dt.datetime,
    timeframe: Timeframe,
    tickers: list[Ticker] | None = None,
):
    await db.connect()

    feed = CandlesFeed(start_at, end_at, timeframe, tickers)
    actual_tickers = await feed.get_actual_tickers()

    manager = WorkManager(actual_tickers, strategy)
    try:
        await manager.start_workers()
        await feed.send_candles()
        await manager.wait_workers()

    finally:
        await manager.shutdown_workers()

    await db.close()


if __name__ == '__main__':
    asyncio.run(
        backtest(
            strategy=TestStrategy(),
            start_at=dt.datetime(2022, 4, 1),
            end_at=dt.datetime(2022, 4, 10),
            timeframe=Timeframe.M5,
            tickers=['BTCUSDT', 'ETHUSDT'],
            # tickers=None,
        ),
    )
