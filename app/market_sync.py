import functools
import logging
import typing as tp

from tortoise import timezone as tz

from dobermann import BinanceClient, Timeframe

from . import models, scheduler

if tp.TYPE_CHECKING:
    from dobermann.binance_client import FuturesAsset

logger = logging.getLogger(__name__)


@scheduler.job()
async def sync_assets():
    async with BinanceClient() as client:
        assets: tp.Dict[str, FuturesAsset] = {a.ticker: a for a in await client.get_futures_assets()}

    api_tickers = set(assets.keys())
    db_tickers = set(
        await models.Asset.filter(removed_at__isnull=True).values_list('ticker', flat=True)
    )

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


@scheduler.job()
async def sync_candles(timeframe: Timeframe):
    ...


sync_candles_5m = functools.partial(sync_candles, Timeframe.M5)


def run():
    # scheduler.add_job(sync_assets, '59 * * * *')
    scheduler.add_job(sync_assets, '* * * * *')
    scheduler.add_job(sync_candles_5m, '*/5 * * * *')

    scheduler.run_scheduler()
