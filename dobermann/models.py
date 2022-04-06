import datetime as dt
import json
import typing as tp
from enum import Enum
import typing as tp

from tortoise import fields, models
from tortoise.queryset import QuerySet

from dobermann.binance_client import Timeframe


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: tp.Any) -> tp.Any:
        if isinstance(obj, (dt.date, dt.datetime)):
            return obj.isoformat()
        return super().default(obj)


class Asset(models.Model):
    class Status(Enum):
        TRADING = 'TRADING'
        PENDING_TRADING = 'PENDING_TRADING'
        BREAK = 'BREAK'

    ticker = fields.CharField(max_length=16, index=True)

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

    @classmethod
    def active(cls) -> tp.Awaitable[QuerySet]:
        return cls.filter(removed_at__isnull=True)


class Candle(models.Model):
    asset = fields.ForeignKeyField('models.Asset', related_name='candles', on_delete=fields.CASCADE, index=True)
    timeframe = fields.CharEnumField(Timeframe, index=True)

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
