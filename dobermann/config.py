import datetime as dt
import typing as tp
import logging
from decimal import Decimal
from pathlib import Path

import pytz
from pydantic import BaseSettings, SecretStr


class Settings(BaseSettings):
    BASE_DIR = Path(__file__).resolve().parent.parent

    ENVIRONMENT: str = 'local'
    assert ENVIRONMENT in {'local', 'prod'}

    DB_HOST: str = 'localhost'
    DB_PORT: int = 5432
    DB_USER: str = 'dobermann'
    DB_NAME: str = 'dobermann'
    DB_PASSWORD: SecretStr = SecretStr('dobermann')

    DB_URI: SecretStr = SecretStr(
        f'postgresql://{DB_USER}:{DB_PASSWORD.get_secret_value()}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )

    TZ_NAME: str = 'Asia/Yekaterinburg'
    TIMEZONE: tp.Any = pytz.timezone(TZ_NAME)

    BINANCE_KEY: SecretStr = SecretStr('')
    BINANCE_SECRET: SecretStr = SecretStr('')
    BINANCE_F_COMISSION: Decimal = Decimal(0.0004)  # Комиссия Binance Futures за одну операцию (в % от суммы сделки)

    TORTOISE_ORM: dict = {
        'connections': {
            'default': {
                'engine': 'tortoise.backends.asyncpg',
                'credentials': {
                    'host': DB_HOST,
                    'port': DB_PORT,
                    'user': DB_USER,
                    'password': DB_PASSWORD.get_secret_value(),
                    'database': DB_NAME,
                },
                'maxsize': 10,
            },
        },
        'apps': {
            'models': {
                'models': ['dobermann.models', 'aerich.models'],
            },
        },
        'use_tz': True,
        'timezone': TZ_NAME,
    }

    LOGGING_LEVEL: str = 'DEBUG'
    LOGGING: dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'class': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(levelname)-8s%(asctime)s%(yellow)s %(name)s: ' '%(blue)s%(message)s %(reset)s',  # noqa
            },
        },
        'handlers': {
            'stdout': {
                'level': LOGGING_LEVEL,
                'class': 'logging.StreamHandler',
                'formatter': 'default',
            },
        },
        'loggers': {
            '': {
                'handlers': ['stdout'],
                'level': LOGGING_LEVEL,
            },
            'asyncio': {
                'level': logging.ERROR,
            },
            'tortoise': {
                'level': logging.ERROR,
            },
            'apscheduler.executors': {
                'level': logging.ERROR,
            },
            'parso': {
                'level': logging.INFO,
            }
        },
    }

    class Config:
        env_file = '.env'
        allow_mutation = False


settings = Settings()

TORTOISE_ORM = settings.TORTOISE_ORM
