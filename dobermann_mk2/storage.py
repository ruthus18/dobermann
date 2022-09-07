import datetime as dt
import typing as tp

from motor.motor_asyncio import AsyncIOMotorClient

if tp.TYPE_CHECKING:
    from pymongo.results import InsertManyResult


from .core import Candle, Asset, Timeframe
from .config import logger
from . import bybit


DB_HOST = 'localhost'
DB_PORT = 27017
DB_USER = 'dobermann'
DB_PASSWORD = 'dobermann'
DB_NAME = 'dobermann'


DB_URI = f'mongodb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?authSource=admin'


# TODO: retries on connection failure in tasks
client = AsyncIOMotorClient(DB_URI)
db = client[DB_NAME]


async def check_connected() -> dict:
    return await client.server_info()


async def insert_candles(candles: list[Candle], asset: Asset, timeframe: Timeframe) -> 'InsertManyResult':
    result = await db.candles.insert_many([
        {
            **candle,
            'asset': asset,
            'timeframe': timeframe,
        }
        for candle in candles
    ])
    logger.info('{} candles inserted for {}', len(result.inserted_ids), asset)

    return result


async def sync_candles_from_bybit():
    absolute_start_at = dt.datetime(2015, 1, 1)
    absolute_end_at = dt.datetime.now()

    assets = bybit.TEST_ASSETS
    timeframes = (Timeframe.D1, Timeframe.H1, Timeframe.M5)

    logger.info('Performing candles sync...')

    for asset in assets:
        for timeframe in timeframes:

            candles = await bybit.client.get_candles(asset, timeframe, absolute_start_at, absolute_end_at)
            await insert_candles(candles, asset, timeframe)

    logger.info('Candles sync is done')
