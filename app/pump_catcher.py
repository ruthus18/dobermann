import logging
import logging.config

from .binance_client import BinanceClient
from .config import settings

logger = logging.getLogger(__name__)

logging.config.dictConfig(settings.LOGGING)


# s - ticker, E - current time, p - current price
async def on_message(prices: dict):
    pass


async def run():
    async with BinanceClient() as client:
        await client.subscribe_market_mark_price(callback=on_message, rate_limit=60)
