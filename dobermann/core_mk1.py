import asyncio
import datetime as dt
import logging.config
import random
import time
import typing as tp
from pprint import pprint
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal

import simplejson as json
import zmq
from tortoise.queryset import QuerySet
from zmq.asyncio import Context, Poller

from . import db, models
from .config import settings

from .binance_client import Timeframe
from .utils import cancel_task, split_list_round_robin

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')


WORKERS_POOL_SIZE = 2

_ZMQ_URL = 'tcp://127.0.0.1:{port}'

# Business logic data flow
CANDLES_URL = _ZMQ_URL.format(port=4444)
SIGNALS_URL = _ZMQ_URL.format(port=4445)
EXCHANGE_URL = _ZMQ_URL.format(port=4446)

# Infrastructure data flow
FEED_REGISTRATOR_URL = _ZMQ_URL.format(port=4447)
WORKERS_CTRL_URL = _ZMQ_URL.format(port=4448)

SESSION_ID = uuid.uuid4().hex
SESSION_TIME = dt.datetime.now().strftime('%y%m%d%H%M%S')


Ticker = str
SignalType = tp.TypeVar('SignalType')


# TODO:
#   * Обработка ситуаций, когда нет свечей для актива в указаном диапазное (start_at ; end_at)


@dataclass
class Candle:
    open_time: dt.datetime
    open: Decimal
    close: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal


def candle_to_json(obj: models.Candle) -> str:
    return json.dumps(
        dict(
            open_time=obj.open_time,
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


class StrategyRuntimeError(Exception): ...


class Strategy(ABC):

    def __init__(self):
        self.ticker: tp.Optional[Ticker] = None
        self.exchange: tp.Optional['ExchangeClient'] = None

        self.candle: tp.Optional[dict] = None
        self.position_id: tp.Optional[str] = None

        self.signals: tp.Dict[dt.datetime: SignalType] = {}

    def _init_worker(self, ticker: Ticker, exchange_client: 'ExchangeClient'):
        self.ticker = ticker
        self.exchange = exchange_client

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

    async def on_candle(self, candle: Candle) -> dict:
        self.candle = candle
        signal = await self.calculate()
        self.signals[candle.open_time] = signal    

    @abstractmethod
    async def calculate(self) -> SignalType: ...


class TestStrategy(Strategy):
    def __init__(self):
        super().__init__()

    async def calculate(self) -> SignalType:
        return random.randint(0, 5)


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
        self.stats = {}

        self._assets: QuerySet[models.Asset] = None
        self._candles_iter: tp.Awaitable[QuerySet[models.Candle]] = None
        self.candles_total = None

        self.zmq_ctx: zmq.Context
        self.candles_sender: zmq.Socket
        self.init_zmq()

    def init_zmq(self):
        self.zmq_ctx = Context.instance()

        self.candles_sender = self.zmq_ctx.socket(zmq.PUB)
        self.candles_sender.bind(CANDLES_URL)
        self.candles_sender.set(zmq.SNDHWM, 0)

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
        _ts = time.time()

        self.assets = await self._get_assets_from_db()
        self.log.debug('Assets loaded: %s', len(self.assets))

        self.candles_iter = models.Candle.filter(
            asset_id__in=[a.id for a in self.assets],
            timeframe=self.timeframe,
            open_time__gte=self.start_at,
            open_time__lt=self.end_at,
        ).select_related('asset').order_by('open_time')

        self.candles_total = await self.candles_iter.count()
        self.log.debug('Total candles: %s', self.candles_total)

        self.stats['prepare_data_time'] = time.time() - _ts
        self.stats['candles_total'] = self.candles_total

    async def get_candles_count(self, tickers: tp.Iterable[Ticker]) -> tp.Dict[Candle, int]:
        # FIXME: one query
        total_candles = {}
        for ticker in tickers:
            total = await self.candles_iter.filter(asset__ticker=ticker).count()
            total_candles[ticker] = total

        return total_candles

    async def wait_actors(self):
        _ts = time.time()

        registrator = self.zmq_ctx.socket(zmq.REP)
        registrator.bind(FEED_REGISTRATOR_URL)

        workers_remaining = self.pool_size
        exchange_ready = False

        self.log.debug('Waiting workers and exchange to register...')

        while workers_remaining or not exchange_ready:
    
            request_type, request_data = await registrator.recv_multipart()
            request_type = request_type.decode()

            if request_type == 'reg_worker':
                tickers = json.loads(request_data.decode())
                response_data = await self.get_candles_count(tickers)

                workers_remaining -= 1

            elif request_type == 'reg_exchange':
                response_data = self.tickers
                exchange_ready = True

            else:
                raise RuntimeError('Unknown request type while registering actors')

            await registrator.send_multipart((request_type.encode(), json.dumps(response_data).encode()))

        self.log.debug('Workers and exchange registered')
        registrator.close()

        self.stats['wait_actors_time'] = time.time() - _ts

    async def run(self):
        self.log.debug('Candles feed started')
        _ts_total = time.time()

        with suppress(asyncio.CancelledError):
            await self.prepare_data()
            await self.wait_actors()

            self.log.debug('Sending candles to workers...')
            _ts_process = time.time()
            async for candle in self.candles_iter:
                await self.candles_sender.send_multipart((candle.asset.ticker.encode(), candle_to_json(candle)))

            self.stats['process_candles_time'] = time.time() - _ts_process

            self.log.debug('Finish sending candles')

            _ts_idle = time.time()
            await asyncio.sleep(float('inf'))  # wait external cancellation

        self.stats['idle_time'] = time.time() - _ts_idle
        self.close_zmq()

        self.stats['total_time'] = time.time() - _ts_total
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

        zmq_ctx = Context.instance()
        workers_controller: zmq.Socket = zmq_ctx.socket(zmq.PUB)
        workers_controller.bind(WORKERS_CTRL_URL)

        loop = asyncio.get_event_loop()
        proc_executor = ProcessPoolExecutor(max_workers=self.pool_size)
        tickers_splitted = split_list_round_robin(self.tickers, self.pool_size)

        with suppress(asyncio.CancelledError):
            workers = [
                loop.run_in_executor(
                    proc_executor, spawn_worker, w_id, self.strategy, tickers_splitted[w_id],
                )
                for w_id in range(self.pool_size)
            ]
            await asyncio.sleep(float('inf'))  # wait external cancellation

        self.log.debug('Shutdown workers...')
        await workers_controller.send_string('exit')

        await asyncio.wait(workers)
        proc_executor.shutdown(wait=True)

        workers_controller.close()
        self.log.debug('Manager stopped')


def spawn_worker(w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
    logger.debug('Worker spawned')
    worker = Worker(w_id, strategy, tickers)
    worker.init_zmq()

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
        # worker.close_zmq()
        worker.log.debug('Worker stopped')


# TODO: Check integrity errors in time series
class Worker:

    def __init__(self, w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
        self.w_id = w_id
        self.tickers = tickers

        self.log = logger.getChild(f'worker[{w_id}]')

        self._strategy = strategy
        self.strategies: tp.Dict[Ticker, Strategy] = {}
        self.candles_remaining: tp.Dict[Ticker, int] = {}

        self.zmq_ctx: zmq.Context

        self.poller: zmq.Poller
        self.registrator: zmq.Socket
        self.controller: zmq.Socket

        self.candle_receiver: zmq.Socket
        self.signal_sender: zmq.Socket

        self.exchange_client: ExchangeClient

    def init_zmq(self):
        self.zmq_ctx = Context.instance()

        self.candle_receiver = self.zmq_ctx.socket(zmq.SUB)
        self.candle_receiver.connect(CANDLES_URL)
        self.candle_receiver.set(zmq.RCVHWM, 0)

        for ticker in self.tickers:
            self.candle_receiver.subscribe(ticker)
            self.log.debug('Subscribed to candle %s', ticker)

        self.controller = self.zmq_ctx.socket(zmq.SUB)
        self.controller.connect(WORKERS_CTRL_URL)
        self.controller.subscribe('')

        self.poller = Poller()
        self.poller.register(self.candle_receiver, zmq.POLLIN)
        self.poller.register(self.controller, zmq.POLLIN)

        self.signal_sender = self.zmq_ctx.socket(zmq.PUSH)
        self.signal_sender.connect(SIGNALS_URL)
        self.signal_sender.set(zmq.SNDHWM, 0)

        self.exchange_client = ExchangeClient(zmq_ctx=self.zmq_ctx)

    def close_zmq(self):
        self.candle_receiver.close()
        self.controller.close()
        self.signal_sender.close()

        self.exchange_client.close()

        self.zmq_ctx.term()

    async def register(self):
        registrator = self.zmq_ctx.socket(zmq.REQ)
        registrator.connect(FEED_REGISTRATOR_URL)

        await registrator.send_multipart((b'reg_worker', json.dumps(self.tickers).encode()))
        response_type, response_data = await registrator.recv_multipart()

        assert response_type == b'reg_worker'
        self.candles_remaining = json.loads(response_data.decode())

        self.log.debug('Registered! Remaining candles: %s', self.candles_remaining)

        registrator.close()

    async def run(self):
        for ticker in self.tickers:
            strategy = deepcopy(self._strategy)
            strategy._init_worker(ticker, self.exchange_client)

            self.strategies[ticker] = strategy

        await self.register()

        self.log.debug('Waiting events...')
        while True:
            sockets = dict(await self.poller.poll())

            if sockets.get(self.candle_receiver) == zmq.POLLIN:
                ticker, candle_data = await self.candle_receiver.recv_multipart()

                ticker = ticker.decode()
                candle = candle_from_json(candle_data)
                # self.log.debug('Received candle %s: %s', ticker, candle)

                strategy = self.strategies[ticker]
                await strategy.on_candle(candle)

                self.candles_remaining[ticker] -= 1

                if self.candles_remaining[ticker] == 0:
                    self.log.debug('All candles for %s processed', ticker)

                    await self.signal_sender.send(json.dumps({
                        'ticker': ticker,
                        'signals': strategy.signals,
                    }).encode())

                    await self.exchange_client.finish_processing(ticker)

            if sockets.get(self.controller) == zmq.POLLIN:
                await self.controller.recv_string()

                self.log.debug('Received exit event')
                return


async def signal_exporter():
    log = logger.getChild('exporter')
    log.debug('Signals exporter finished')

    zmq_ctx = Context.instance()
    signals_receiver = zmq_ctx.socket(zmq.PULL)
    signals_receiver.bind(SIGNALS_URL)
    signals_receiver.set(zmq.RCVHWM, 0)

    total = 4
    with suppress(asyncio.CancelledError):
        while True:
            data = await signals_receiver.recv()
            event = json.loads(data.decode())
            log.warning('Received event: %s (%s)', event['ticker'], len(event['signals']))

            total -= 1
            if total == 0:
                break

    signals_receiver.close()
    log.debug('Signals exporter finished')


# filename = f'{SESSION_ID}_{SESSION_TIME}_signals.csv'


class ExchangeClient:

    def __init__(self, zmq_ctx: zmq.Context):
        self.zmq_ctx = zmq_ctx

        self.sender: zmq.Socket = self.zmq_ctx.socket(zmq.PUSH)
        self.sender.connect(EXCHANGE_URL)

    def close(self):
        self.sender.close()

    async def notify_event(self, event_name: str, event_data: dict):
        await self.sender.send_multipart((event_name.encode(), json.dumps(event_data).encode()))

    async def open_position(self, data: dict):
        await self.notify_event('open_position', data)

    async def close_position(self, data: dict):
        await self.notify_event('close_position', data)

    async def finish_processing(self, ticker: str):
        await self.notify_event('finish_processing', {'ticker': ticker})


class Exchange:

    def __init__(self):
        self.log = logger.getChild('exchange')
        self.zmq_ctx = Context.instance()

        self.remaining_tickers: tp.Set[Ticker]

    async def register(self):
        registrator = self.zmq_ctx.socket(zmq.REQ)
        registrator.connect(FEED_REGISTRATOR_URL)        

        await registrator.send_multipart((b'reg_exchange', b''))
        _, data = await registrator.recv_multipart()
        self.remaining_tickers = set(json.loads(data.decode()))

        self.log.debug('Registered! Total tickers: %s', len(self.remaining_tickers))

        registrator.close()

    async def run(self):
        self.log.debug('Exchange started')

        exchange_sock = self.zmq_ctx.socket(zmq.PULL)
        exchange_sock.bind(EXCHANGE_URL)

        with suppress(asyncio.CancelledError):
            await self.register()

            self.log.debug('Waiting events...')
            while True:
                event_name, event_data = await exchange_sock.recv_multipart()
                event_name = event_name.decode()
                event_data = json.loads(event_data.decode())

                self.log.debug('Got event `%s`: %s', event_name, event_data)

                if event_name == 'open_position':
                    ...

                elif event_name == 'close_position':
                    ...

                elif event_name == 'finish_processing':
                    self.remaining_tickers.remove(event_data['ticker'])

                    if len(self.remaining_tickers) == 0:
                        self.log.debug('All data processed, returning')
                        break

                else:
                    raise RuntimeError(f'unknown event name: `{event_name}`')

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
    t_sig_exporter = asyncio.create_task(signal_exporter())

    try:
        await asyncio.wait((t_exchange, ))

    except (KeyboardInterrupt, SystemExit):
        pass

    finally:
        logger.debug('Waiting tasks to complete...')

        await cancel_task(t_manager)
        await cancel_task(t_feed)
        await cancel_task(t_exchange)
        await cancel_task(t_sig_exporter)

        logger.info('Trading system is stopped, goodbye 👋')


async def backtest(
    strategy: Strategy,
    start_at: dt.datetime,
    end_at: dt.datetime,
    timeframe: Timeframe,
    tickers: tp.Optional[tp.List[Ticker]],
):
    await db.connect()

    if not start_at.tzinfo:
        start_at = start_at.astimezone(settings.TIMEZONE)

    if not end_at.tzinfo:
        end_at = end_at.astimezone(settings.TIMEZONE)

    if tickers is None:
        tickers = await models.Asset.filter(removed_at__isnull=True).values_list('ticker', flat=True)

    pool_size = min(WORKERS_POOL_SIZE, len(tickers))

    candles_feed = CandlesFeed(start_at, end_at, timeframe, tickers, pool_size)
    manager = Manager(strategy, tickers, pool_size)
    exchange = Exchange()
    await run_trading_system(candles_feed, manager, exchange)

    pprint(candles_feed.stats)

    await db.close()


if __name__ == '__main__':
    asyncio.run(
        backtest(
            strategy=TestStrategy(),
            start_at=dt.datetime(2022, 3, 1),
            end_at=dt.datetime(2022, 3, 10),
            timeframe=Timeframe.M5,
            tickers=['BTCUSDT', 'ETHUSDT'],
            # tickers=None,
        ),
    )
