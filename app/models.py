import datetime as dt
import json
import typing as tp
from enum import Enum
from typing import Any, Dict, Sequence

from tortoise import Tortoise, fields, models

from dobermann.binance_client import Timeframe

from .config import settings


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (dt.date, dt.datetime)):
            return obj.isoformat()
        return super().default(obj)


async def init_db() -> None:
    await Tortoise.init(settings.TORTOISE_ORM)


async def close_db() -> None:
    await Tortoise.close_connections()


async def db_query(sql: str) -> Sequence[Dict[Any, Any]]:
    conn = Tortoise.get_connection("default")
    _, result = await conn.execute_query(sql)

    return result


async def truncate(model: models.Model):
    return await db_query(f'TRUNCATE TABLE {model._meta.db_table}')


async def db_size() -> str:
    result = await db_query(f"SELECT pg_database_size('{settings.DB_NAME}')/1024 AS kb_size;")

    size_mb = result[0].get('kb_size') / 1024  # type: ignore
    if size_mb // 1024 == 0:
        return f'{size_mb:.3f} MB'

    size_gb = size_mb / 1024
    return f'{size_gb:.3f} GB'


class Asset(models.Model):
    class Status(Enum):
        TRADING = 'TRADING'
        PENDING_TRADING = 'PENDING_TRADING'
        BREAK = 'BREAK'

    ticker = fields.CharField(max_length=16, index=True, unique=True)

    base_asset = fields.CharField(max_length=16)
    quote_asset = fields.CharField(max_length=16)
    status = fields.CharEnumField(Status)
    onboard_date = fields.DateField()
    filters = fields.JSONField(encoder=JsonEncoder().encode)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(null=True)
    removed_at = fields.DatetimeField(null=True)

    def __str__(self):
        return self.ticker

    # TODO: move to queryset
    @classmethod
    def active(cls) -> tp.Iterable[str]:
        return cls.filter(removed_at__isnull=True)


class Candle(models.Model):
    asset = fields.ForeignKeyField('models.Asset', related_name='candles', on_delete=fields.CASCADE)
    timeframe = fields.CharEnumField(Timeframe)

    open_time = fields.DatetimeField()
    open = fields.DecimalField(max_digits=16, decimal_places=8)
    close = fields.DecimalField(max_digits=16, decimal_places=8)
    low = fields.DecimalField(max_digits=16, decimal_places=8)
    high = fields.DecimalField(max_digits=16, decimal_places=8)
    volume = fields.DecimalField(max_digits=24, decimal_places=8)

    class Meta:
        unique_together = (('asset', 'timeframe', 'open_time'), )

    def __str__(self):
        return f'[{self.asset.ticker} {self.timeframe}] {self.open_time}'
