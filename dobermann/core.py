import asyncio
from dataclasses import dataclass
from decimal import Decimal
import datetime as dt
import logging.config
import signal
import time
import typing as tp
from abc import ABC, abstractmethod

from tortoise.queryset import QuerySet
import zmq
from zmq.asyncio import Context

from app.config import settings
from app import models
from app import db

from .base import Strategy, Ticker, Timeframe
from .utils import cancel_task

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')


LOGS_URL = 'tcp://127.0.0.1:4567'
FEED_URL = 'tcp://127.0.0.1:4568'


class Actor(ABC):
    logging_name = NotImplemented

    def __init__(self):
        self.zmq_ctx = Context.instance()
        self._logs_sock = self.zmq_ctx.socket(zmq.PUSH)

    async def log(self, message: str):
        await self._logs_sock.send_string(f'{self.logging_name};{message}')

    async def setup(self):
        self._logs_sock.connect(LOGS_URL)
        await self.log('Actor connected')

    async def shutdown(self):
        await self.log('Actor disconnected')
        await asyncio.sleep(0.1)
        self._logs_sock.close()

    @abstractmethod
    async def run(self): ...


class MarketFeed(Actor):
    """Base class for supplying market data to trading system.
    """
    logging_name = 'feed'


@dataclass
class CandleEvent:
    ticker: Ticker
    timeframe: Timeframe
    open_time: dt.datetime
    open: Decimal
    close: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal

    @classmethod
    def from_db_model(cls, model: models.Candle) -> 'CandleEvent':
        ...

    @classmethod
    def from_msg(cls, message: bytes) -> 'CandleEvent':
        ...

    def to_msg(self) -> bytes:
        ...


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
        return await super().shutdown()

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
            await self.log(f'Got candle {candle.asset.ticker} {candle.open_time}')

        await asyncio.sleep(float('inf'))  # wait cancellation


async def logs_watcher(logs_sock: zmq.Socket) -> None:
    while True:
        event = await logs_sock.recv_string()
        name, msg = event.split(';')

        logger.getChild(name).info(msg)


async def run_trading_system(
    market_feed: MarketFeed,
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

    logger_task = asyncio.create_task(logs_watcher(logs_sock))
    feed_task = asyncio.create_task(market_feed.run())

    try:   
        await stop_event.wait()
        logger.info('Received stop event, shutdown...')

    finally:
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

    logger.info('Backtesting started')

    elapsed = time.time()
    await run_trading_system(
        market_feed=market_feed,
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
