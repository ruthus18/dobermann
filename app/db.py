import datetime as dt
import pathlib
import typing as t
from contextlib import asynccontextmanager

import asyncpg

from . import bybit, config
from .config import logger
from .core import Asset, Candle, Timeframe

pool: asyncpg.pool.Pool


async def start() -> None:
    global pool
    pool = await asyncpg.create_pool(config.DB_URI)


async def close() -> None:
    await pool.close()


@asynccontextmanager
async def connect() -> t.AsyncIterator[asyncpg.Connection]:
    async with pool.acquire(timeout=config.DB_CONNECT_TIMEOUT) as conn:
        yield conn


@asynccontextmanager
async def transaction() -> t.AsyncIterator[asyncpg.Connection]:
    async with connect() as conn:
        async with conn.transaction():
            yield conn


async def apply_migrations() -> None:
    async with connect() as conn:
        await conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS migrations (
                name TEXT UNIQUE NOT NULL,
                applied_at TIMESTAMP NOT NULL DEFAULT current_timestamp
            )
            '''
        )

    async with transaction() as conn:
        await conn.execute('LOCK TABLE migrations IN EXCLUSIVE MODE')

        migrations_dir = pathlib.Path(__file__).parent.joinpath('migrations')
        for path in sorted(migrations_dir.glob('*.sql')):
            name, sql = path.stem, path.read_text()
            assert sql, f'File {name} is empty'

            if await conn.fetchval('SELECT 1 FROM migrations WHERE name = $1', name):
                logger.debug('migration %s exists', name)
                continue

            logger.info('applying migration %s', name)
            async with conn.transaction():  # Savepoint
                await conn.execute(sql)
                await conn.execute('INSERT INTO migrations (name) VALUES ($1)', name)


async def _set_numeric_to_float_conversion_codec(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        'numeric',
        encoder=str,
        decoder=float,
        format='text',
        schema='pg_catalog',
    )


async def insert_candles(candles: list[Candle], asset: Asset, timeframe: Timeframe) -> None:
    values = (
        (asset, timeframe, c['open_time'], c['open'], c['close'], c['low'], c['high'], c['volume'])
        for c in candles
    )
    async with connect() as conn:
        await conn.executemany(
            '''
            INSERT INTO candles (asset, timeframe, open_time, open, close, low, high, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
            ''',
            values
        )

    logger.info('{} candles inserted for <g>{}[{}]</g>', len(candles), asset, timeframe)


async def get_candles(
    asset: Asset, timeframe: Timeframe, start_at: dt.datetime, end_at: dt.datetime
) -> list[Candle]:
    candles = []

    async with transaction() as conn:
        await _set_numeric_to_float_conversion_codec(conn)
        cursor = await conn.cursor(
            '''
            SELECT open_time, open, close, low, high, volume
            FROM candles
            WHERE asset = $1
                AND timeframe = $2
                AND open_time >= $3
                AND open_time < $4
            ORDER BY open_time;
            ''',
            *(asset, timeframe, start_at, end_at)
        )
        while candles_batch := await cursor.fetch(1000):
            candles += [Candle(record) for record in candles_batch]

    return candles


# TODO: If candles exist in DB -> start sync from (max(open_time) + timeframe.timedelta)
async def sync_candles_from_bybit() -> None:
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
