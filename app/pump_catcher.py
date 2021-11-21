import datetime as dt
import logging
import logging.config

import dramatiq
from dramatiq.brokers.redis import RedisBroker

from .binance_client import BinanceClient
from .config import settings
from .trading import RoundedDecimal

logger = logging.getLogger(__name__)

logging.config.dictConfig(settings.LOGGING)


redis_broker = RedisBroker(host='127.0.0.1', port='6379', password='dobermann')
dramatiq.set_broker(redis_broker)


# s - ticker, E - current time, p - current price
async def on_message(prices: dict):
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
        await client.subscribe_market_mark_price(callback=on_message, rate_limit=60)
