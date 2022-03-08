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

WORKERS_URL = 'tcp://127.0.0.1:4444'
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
                open_time__lt=self.end_at,
            ).select_related('asset').order_by('open_time')

            self.log.debug('Sending candles to workers...')
            self.log.debug('Total candles: %s', await candles_iter.count())

            await asyncio.sleep(3)  # TODO: Убрать это говнище

            await feed_sock.send_string(f'last_at;{self.end_at.isoformat()}')

            async for candle in candles_iter:
                json_msg = candle_to_json(candle)

                await feed_sock.send_string(f'{candle.asset.ticker};{json_msg}')

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
        workers_sock.bind(WORKERS_URL)

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
        await workers_sock.send_string('0')

        await asyncio.wait(tasks)
        pool.shutdown(wait=True)

        self.log.debug('Manager stopped')


def spawn_worker(w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
    worker = Worker(w_id, strategy, tickers)
    try:
        asyncio.run(worker.run())

    except (KeyboardInterrupt, SystemExit):
        worker.log.debug('Force shutdown worker...')

    except Exception as e:
        worker.log.exception(
            'Exception occured while processing candles: (%s) %s',
            e.__class__.__name__,
            e,
        )


# FIXME: Нужно переделать формат входящих событий и вынести сообщения в отдельный объект (CandleEvent):
#        <ticker>;<action>;<data>
#        ------------------------
#        BTCUSDT;last_open_at;2022-03-10T06:20:56.915248
#        ETHUSDT;candle;{...}
class Worker:

    def __init__(self, w_id: int, strategy: Strategy, tickers: tp.List[Ticker]):
        self.w_id = w_id
        self.tickers = tickers

        self.log = logger.getChild(f'worker[{w_id}]')

        self._strategy = strategy
        self.strategies: tp.Dict[Ticker, Strategy] = {}
        self.last_at: tp.Optional[dt.datetime] = None

    async def run(self):
        self.log.debug('Worker started')

        _zmq_ctx = Context.instance()
        listener_sock = _zmq_ctx.socket(zmq.SUB)
        listener_sock.connect(FEED_URL)
        listener_sock.connect(WORKERS_URL)

        exchange_sock = _zmq_ctx.socket(zmq.PUSH)
        exchange_sock.connect(EXCHANGE_URL)

        for ticker in self.tickers:
            listener_sock.setsockopt_string(zmq.SUBSCRIBE, ticker)

            strategy = deepcopy(self._strategy)
            strategy._init_worker(ticker, exchange_sock)

            self.strategies[ticker] = strategy
            self.log.debug('Subscribed to candle %s', ticker)

        listener_sock.setsockopt_string(zmq.SUBSCRIBE, '0')  # shutdown event
        listener_sock.setsockopt_string(zmq.SUBSCRIBE, 'last_at')  # last time register event

        self.log.debug('Waiting events...')
        while True:
            msg = await listener_sock.recv_string()
            self.log.debug('Got event: %s', msg)

            if msg == '0':
                break

            elif 'last_at' in msg:
                _, last_at = msg.split(';')
                self.last_at = dt.datetime.fromisoformat(last_at)
                self.log.debug('Register last date: %s', last_at)
                continue

            ticker, candle_data = msg.split(';')

            strategy = self.strategies[ticker]
            candle = candle_from_json(candle_data)

            # TODO: check integrity errors in time series
            await strategy.on_candle(candle)

            if self.last_at and dt.datetime.fromisoformat(candle.close_time) > self.last_at:  # FIXME
                self.log.debug('Handle all candles, exit...')
                break

        listener_sock.close()
        exchange_sock.close()

        self.log.debug('Worker stopped')
        # TODO: Сохранение сигналов стратегий (передавать через сокет в менеджер, который будет писать в csv)


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
            [t_manager, t_feed, t_exchange],
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
            start_at=dt.datetime(2021, 9, 1),
            end_at=dt.datetime(2021, 9, 2),
            timeframe=Timeframe.H1,
            tickers=['BTCUSDT', 'ETHUSDT', 'DYDXUSDT'],
        )
    )
