import asyncio
import datetime as dt
import logging.config
import typing as tp
from functools import partial

from ..dobermann import db, models
from .config import settings

from tortoise import timezone as tz
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from dobermann import BinanceClient, Timeframe

from . import scheduler

if tp.TYPE_CHECKING:
    from dobermann import FuturesAsset

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger('sync')


# TODO: update asset status
@scheduler.job()
async def update_assets():
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


async def update_historical_candles(timeframe: Timeframe = Timeframe.D1):
    last_candle_mark = {
        asset_id: max_time
        for asset_id, max_time in await db.query(
            f"SELECT asset_id, max(close_time) FROM candle WHERE timeframe = '{timeframe}' GROUP BY asset_id;"
        )
    }
    assets = await models.Asset.filter(removed_at__isnull=True).only('id', 'ticker', 'onboard_date')

    async with BinanceClient() as client:
        for asset in tqdm(assets):
            last_candle_at = last_candle_mark.get(asset.id) or dt.datetime.combine(asset.onboard_date, dt.time())

            logger.info('Updating candles for %s from %s', asset.ticker, last_candle_at)

            candles = [
                candle async for candle in tqdm_async(client.get_futures_historical_candles(
                    ticker=asset.ticker,
                    timeframe=timeframe,
                    start=last_candle_at,
                ))
            ]
            await models.Candle.bulk_create([
                models.Candle(
                    asset=asset,
                    timeframe=timeframe,
                    **candle.dict()
                )
                for candle in candles
            ])
            logger.info('Created %s candles for %s', len(candles), asset.ticker)

    logger.info('Done')


# <DRAFT> (for live trading purposes only, on backtest this logic not using)

# timeframe_to_delta = {
#     Timeframe.M1: dt.timedelta(minutes=1),
# }


# @scheduler.job()
# async def sync_candles(timeframe: Timeframe):
#     open_time = tz.now().replace(second=0, microsecond=0) - timeframe_to_delta[timeframe]
#     assets = await models.Asset.active().only('id', 'ticker')

#     async with BinanceClient() as client:

#         async def coro(asset: models.Asset) -> models.Candle:
#             candle = [
#                 # TODO: Получать время свечи исходя из tz.now()
#                 c for c in await client.get_recent_candles(asset.ticker, timeframe)
#                 if c.open_time == open_time
#             ][0]

#             await models.Candle.create(
#                 asset=asset, timeframe=timeframe, **candle.dict()
#             )

#             return candle

#         candles = await asyncio.gather(coro(asset) for asset in assets)

#     logger.info('Total candles: %s, timestamp: %s', len(candles), candles[0].open_time)


# MIN_CANDLE_TIME = dt.datetime(2010, 1, 1)


# @scheduler.job()
# async def load_candles(timeframe: Timeframe):
#     """Загрузить свечи для всех активов на выбранном таймфрейме.
    
#     Функция загружает исторические свечи, которые отсутствуют в базе данных.
#     """
#     sql = f'''
#         SELECT asset.id, asset.ticker, max(candle.open_time) FROM asset
#             LEFT JOIN candle ON asset.id = candle.asset_id
#         WHERE candle.timeframe = '{timeframe}' GROUP BY asset.id;
#     '''
#     asset_2_last_candles_at = await db.query(sql)
#     print(len(asset_2_last_candles_at))

#     now = tz.now()

#     for asset_id, ticker, last_candle_time in asset_2_last_candles_at:
#         start_at = last_candle_time or MIN_CANDLE_TIME
#         end_at = now

#         logger.info('Fetching candles for %s (%s - %s)', ticker, start_at, end_at)

#         candles = [candle async for candle in tqdm(client.get_futures_historical_candles(
#             ticker=ticker, timeframe=timeframe, start=start_at, end=end_at
#         ))]

#         await models.Candle.bulk_create([models.Candle(
#             asset_id=asset_id, timeframe=timeframe, **candle.dict()
#         )] for candle in candles)
#         logger.info('Created %s candles for %s', len(candles), ticker)


# TODO: Раздельно конфигурировать lab и live среду
# def main():
#     scheduler.add_job(
#         sync_assets,
#         '58 * * * *',  # every hour before candles sync
#     )
#     scheduler.add_job(
#         partial(sync_candles, Timeframe.M1),
#         '* * * * *',  # every 5 minutes
#         name='sync_candles[1m]',
    # )
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

    # asyncio.run(scheduler.run_scheduler())

# </DRAFT>


async def main():
    await db.init()
    logger.info('Updating 1H candles...')
    await update_historical_candles(Timeframe.H1)

    logger.info('Updating 1H candles...')
    await update_historical_candles(Timeframe.M5)
    await db.close()


if __name__ == '__main__':
    asyncio.run(main())
