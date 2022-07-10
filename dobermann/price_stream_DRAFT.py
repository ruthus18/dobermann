"""
This module is draft of real-time price tracking for futures

Usage:
    1. Start redis server
        ~$ docker-compose up -d redis

    2. Launch socket listenner for price stream
        ~$ ipython

        >>> from app import price_stream
        >>> await price_stream.run()

    3. Launch workers for processing stream data
        ~$ dramatiq app.price_stream
"""

import datetime as dt
import logging
import logging.config

import dramatiq
from dramatiq.brokers.redis import RedisBroker

from dobermann import BinanceClient
from dobermann.utils import RoundedDecimal

from .config import settings

logger = logging.getLogger(__name__)

logging.config.dictConfig(settings.LOGGING)


redis_broker = RedisBroker(host='127.0.0.1', port='6379', password='dobermann')
dramatiq.set_broker(redis_broker)


# s - ticker, E - current time, p - current price
async def on_new_prices(prices: dict):
    logger.info(prices[0])
    dramatiq.group([
        process_current_price.message(
            ticker=item['s'],
            timestamp=int(item['E'] / 1000),
            price=item['p'],
        ) for item in prices
    ]).run()


@dramatiq.actor
def process_current_price(ticker: str, timestamp: int, price: float):
    logger.info('%s %s %s', ticker, dt.datetime.fromtimestamp(timestamp), RoundedDecimal(price))


async def run():
    async with BinanceClient() as client:
        await client.subscribe_market_mark_price(callback=on_new_prices, rate_limit=15)


async def backtest():
    async with BinanceClient() as client:
        historic_candles = await client.get_futures_historical_candles(...)

    for prices in historic_candles:
        await on_new_prices(prices)
