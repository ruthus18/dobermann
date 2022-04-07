from abc import ABC, abstractmethod
from contextlib import suppress
import asyncio
import datetime as dt
import logging.config
import typing as tp

from .config import settings
from .binance_client import Timeframe

from . import db

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('core')  # TODO: Затянуть либу loguru


Ticker = str
AssetID = int
SignalType = tp.TypeVar('SignalType')


_ZMQ_URL = 'tcp://127.0.0.1:{port}'

# Business logic data flow
# ------------------------
# Feed (PUB) -> Workers (SUB) : candles & stop events
CANDLES_URL = _ZMQ_URL.format(port=22222)
# Workers (PUSH) -> Exchange (PULL): orders & start/stop events
ORDERS_URLS = _ZMQ_URL.format(port=22223)

# Infrastructure data flow
# ------------------------
# Feed (REQ) -> Manager (REP) : register assets
FEED_REGISTRY_URL = _ZMQ_URL.format(port=22333)
# Manager (PUSH) -> Workers (PULL): register assets requests
MANAGER_REGISTRY_URL = _ZMQ_URL.format(port=22334)
# Workers (PUSH) -> Manager (PULL): register assets responses
WORKERS_REGISTRY_URL = _ZMQ_URL.format(port=22335)


class StrategyRuntimeError(Exception): ...


class Strategy(ABC): ...


class TestStrategy(Strategy): ...


class CandlesFeed:

    def __init__(
        self,
        start_at: dt.datetime,
        end_at: dt.datetime,
        timeframe: Timeframe,
        tickers: tp.Iterable[Ticker] | None,
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

        self.log = logger.getChild('feed')

    async def get_actual_tickers(self) -> tp.Set[Ticker]:
        query = f'''
        SELECT asset.ticker FROM asset
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

        async with db.cursor(query, *args) as cursor:
            tickers = await cursor.fetch(10 ** 4)

        return {t[0] for t in tickers}

    async def wait_assets_registration(self):
        ...

    async def run(self):
        self.log.debug('Candles feed started')

        with suppress(asyncio.CancelledError):
            ...

        self.log.debug('Candles feed stopped')


async def backtest(
    strategy: Strategy,
    start_at: dt.datetime,
    end_at: dt.datetime,
    timeframe: Timeframe,
    tickers: tp.List[Ticker] | None = None,
):
    await db.connect()

    feed = CandlesFeed(start_at, end_at, timeframe, tickers)

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
