import enum
from typing import Any, Dict, Sequence

from tortoise import Tortoise, models

from .config import settings


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
    return f'{size_mb:.3f} MB'


class PositionDirection(str, enum.Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'

    def __str__(self) -> str:
        return self.value


# class Position(models.Model):
#     id = fields.IntField(pk=True)
#     symbol = fields.CharField(max_length=16)
#     direction = fields.CharEnumField(PositionDirection)

#     opened_at = fields.DatetimeField()
#     closed_at = fields.DatetimeField(null=True)
#     open_price = fields.DecimalField(max_digits=24, decimal_places=8)
#     close_price = fields.DecimalField(max_digits=24, decimal_places=8)
#     amount = fields.DecimalField(max_digits=32, decimal_places=8)

#     @property
#     def profit_ratio(self) -> Decimal:
#         if self.direction == PositionDirection.LONG:
#             return self.close_price / self.open_price

#         return self.open_price / self.close_price

#     @property
#     def profit(self) -> Decimal:
#         if self.direction == PositionDirection.LONG:
#             diff = self.close_price - self.open_price

#         else:
#             diff = self.close_price - self.open_price

#         return diff * self.amount


# class BalancePayment(models.Model):
#     id = fields.IntField(pk=True)

#     paid_at = fields.DatetimeField()
#     amount = fields.DecimalField(max_digits=16, decimal_places=2)
