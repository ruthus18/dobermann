import asyncio
from contextlib import asynccontextmanager
import datetime as dt
import logging.config
import signal
import typing as tp
from abc import ABC, abstractmethod

import zmq
from zmq.asyncio import Context

from app.config import settings

from .base import Strategy, Ticker, Timeframe

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')


LOGS_URL = 'tcp://127.0.0.1:4446'
EVENTS_URL = 'tcp://127.0.0.1:4445'


class Actor(ABC):
    logging_name = NotImplemented

    def __init__(self):
        self._zmq_ctx = Context.instance()
        self._logs_sock = self._zmq_ctx.socket(zmq.PUSH)

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
    logging_name = 'feed'
    
    async def run(self):
        while True:
            await asyncio.sleep(5)


async def logs_watcher(logs_sock: zmq.Socket) -> None:
    while True:
        event = await logs_sock.recv_string()
        name, msg = event.split(';')

        logger.getChild(name).info(msg)


@asynccontextmanager
async def cancel_task(task: asyncio.Task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        yield


async def run_trading_system(market_feed: MarketFeed):
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
    assets: tp.List[Ticker],
    timeframes: tp.List[Timeframe],
    start_at: dt.datetime,
    end_at: dt.datetime,
) -> ...:
    pass


if __name__ == '__main__':
    asyncio.run(
        run_trading_system(
            MarketFeed(),
        )
    )
