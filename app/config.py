import sys
from decimal import Decimal

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="[{time:HH:mm:ss.SSS}] <lvl>{message}</lvl>", colorize=True)
logger = logger.opt(colors=True)


DB_HOST = 'localhost'
DB_PORT = 5432
DB_USER = 'dobermann'
DB_PASSWORD = 'dobermann'
DB_NAME = 'dobermann'

DB_URI = f'postgres://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
DB_CONNECT_TIMEOUT = 5


BYBIT_TAKER_DERIVATIVES_FEE = Decimal(0.0006)
