import asyncio
import datetime as dt
import logging.config
import typing as tp
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal

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


EVENT_ALL_CANDLES_SENT = b'-1'


class StrategyRuntimeError(Exception): ...


class Strategy(ABC): ...


class TestStrategy(Strategy): ...


@dataclass(slots=True)
class Candle:
    timeframe: Timeframe
    open_time: dt.datetime
    open: Decimal
    close: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal


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
        self._assets: tp.Dict[Ticker, AssetID] = None

        self.log = logger.getChild('feed')

        self.zmq_ctx = Context.instance()

        self.candles_sender = self.zmq_ctx.socket(zmq.PUB)
        self.candles_sender.bind(CANDLES_URL)
        self.candles_sender.set(zmq.SNDHWM, 0)

        self._total_candles_sent = 0

    async def get_assets(self) -> tp.Dict[AssetID, Ticker]:
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

    async def get_actual_tickers(self) -> tp.Set[Ticker]:
        assets = await self.get_assets()
        return set(assets.values())

    async def register_candle_recipients(self):
        tickers = await self.get_actual_tickers()

        registrator = self.zmq_ctx.socket(zmq.REQ)
        registrator.connect(FEED_REGISTRY_URL)

        self.log.debug('Waiting to register candle recipients...')

        await registrator.send_multipart([t.encode() for t in tickers])
        await registrator.recv()

        self.log.debug('Candle recipients registered')
        registrator.close()

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
            await utils.disable_decimal_conversion_codec(conn)

            async with conn.transaction():
                cursor = await conn.cursor(query, *args)

                while candles_batch := await cursor.fetch(1000):
                    for candle in candles_batch:

                        ticker = assets[candle['asset_id']].encode()
                        await self.candles_sender.send_multipart((ticker, utils.packb_candle(candle)))

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
