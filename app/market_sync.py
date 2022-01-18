import asyncio
import datetime as dt
import logging
import typing as tp
from functools import partial

from tortoise import timezone as tz

from dobermann import BinanceClient, Timeframe

from . import models, scheduler

if tp.TYPE_CHECKING:
    from dobermann import FuturesAsset

logger = logging.getLogger(__name__)


# TODO: update asset status
@scheduler.job()
async def sync_assets():
    async with BinanceClient() as client:
        assets: tp.Dict[str, FuturesAsset] = {a.ticker: a for a in await client.get_futures_assets()}

    api_tickers = set(assets.keys())
    db_tickers = set(await models.Asset.active().values_list('ticker', flat=True))

    to_create_tickers = api_tickers - db_tickers
    to_remove_tickers = db_tickers - api_tickers

    # В случае, если сперва актив был удален из API, а затем снова добавлен -- мы получим ошибку UnqiueConstraint.
    # И нужно будет дорабатывать сохранение таких "моргающих" активов
    await models.Asset.bulk_create([
        models.Asset(
            **assets[ticker].dict()
        )
        for ticker in to_create_tickers
    ])

    await models.Asset.filter(ticker__in=to_remove_tickers).update(removed_at=tz.now())

    logger.info('Assets added: %s, removed: %s', len(to_create_tickers), len(to_remove_tickers))


timeframe_to_delta = {
    Timeframe.M1: dt.timedelta(minutes=1),
}


@scheduler.job()
async def sync_candles(timeframe: Timeframe):
    open_time = tz.now().replace(second=0, microsecond=0) - timeframe_to_delta[timeframe]
    assets = await models.Asset.active().only('id', 'ticker')

    async def coro(asset: models.Asset, client: BinanceClient) -> models.Candle:
        candle = [
            c for c in await client.get_recent_candles(asset.ticker, timeframe)
            if c.open_time == open_time
        ][0]

        await models.Candle.create(
            asset=asset, timeframe=timeframe, **candle.dict()
        )

        return candle

    async with BinanceClient() as client:
        tasks = [coro(asset, client) for asset in assets]
        candles = await asyncio.gather(*tasks)

    logger.info('Total candles: %s, timestamp: %s', len(candles), candles[0].open_time)


@scheduler.job()
async def fix_gaps_in_candles():
    ...


def main():
    scheduler.add_job(
        sync_assets,
        '59 0 * * *',  # everyday at 0:59
    )
    scheduler.add_job(
        partial(sync_candles, Timeframe.M1),
        '* * * * *',  # every 5 minutes
        name='sync_candles[1m]',
    )
    # scheduler.add_job(
    #     partial(sync_candles, Timeframe.M5),
    #     '*/5 * * * *',  # every 5 minutes
    #     name='sync_candles[5m]',
    # )
    # scheduler.add_job(
    #     partial(sync_candles, Timeframe.M15),
    #     '*/15 * * * *',  # every 15 minutes
    #     name='sync_candles[15m]',
    # )
    # scheduler.add_job(
    #     partial(sync_candles, Timeframe.H1),
    #     '0 * * * *',  # every hour
    #     name='sync_candles[1h]',
    # )
    # scheduler.add_job(sync_candles_4h, '0 */4 * * *')                 # every 4 hours
    # scheduler.add_job(sync_candles_1d, '0 0 * * *')         # every day  # TODO: may be timezone mismatch, need check

    # Запускается после полной актуализации свечей (на всех таймфреймах) раз в день
    # scheduler.add_job(fix_gaps_in_candles, '0 4 * * *')

    asyncio.run(scheduler.run_scheduler())
