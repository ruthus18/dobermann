import logging
from decimal import Decimal
from typing import Any, Dict, Literal, Optional

import pytz
from pydantic import BaseSettings, Field, SecretStr, root_validator
from pytz.tzinfo import DstTzInfo


class Settings(BaseSettings):
    ENVIRONMENT: Literal['local', 'prod'] = 'local'

    BINANCE_KEY: SecretStr = SecretStr('')
    BINANCE_SECRET: SecretStr = SecretStr('')

    # Комиссия Binance Futures за одну операцию (от суммы сделки)
    BINANCE_COMISSION: Decimal = Decimal(0.0004)

    DB_HOST: str = 'localhost'
    DB_PORT: int = 5432
    DB_USER: str = 'dobermann'
    DB_PASSWORD: str = 'dobermann'
    DB_NAME: str = 'dobermann'

    TORTOISE_ORM: Dict[str, Any] = {}

    TZ_NAME: str = 'Asia/Yekaterinburg'
    TIMEZONE: Any = None

    LOGGING_LEVEL: Literal['DEBUG', 'INFO', 'ERROR'] = Field('INFO')
    LOGGING: Dict[str, Any] = {}

    @root_validator
    @classmethod
    def post_init(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values['TORTOISE_ORM'].update({
            'connections': {
                'default': {
                    'engine': 'tortoise.backends.asyncpg',
                    'credentials': {
                        'host': values['DB_HOST'],
                        'port': values['DB_PORT'],
                        'user': values['DB_USER'],
                        'password': values['DB_PASSWORD'],
                        'database': values['DB_NAME'],
                    },
                    'maxsize': 10,
                },
            },
            'apps': {
                'models': {
                    'models': ['app.models', 'aerich.models'],
                },
            },
            'use_tz': True,
            'timezone': 'Asia/Yekaterinburg'
        })
        values['LOGGING'].update({
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
                    'level': values['LOGGING_LEVEL'],
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['stdout'],
                    'level': values['LOGGING_LEVEL'],
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
        })

        if not values['TZ_NAME']:
            raise RuntimeError('Timezone name not specified')

        values['TIMEZONE'] = pytz.timezone(values['TZ_NAME'])

        return values

    class Config:
        env_file = '.env'
        allow_mutation = False


settings = Settings()

TORTOISE_ORM = settings.TORTOISE_ORM
