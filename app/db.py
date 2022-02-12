import typing as tp
from tortoise import Tortoise

from .config import settings

from tortoise import models


async def init() -> None:
    await Tortoise.init(settings.TORTOISE_ORM)


async def close() -> None:
    await Tortoise.close_connections()


async def query(sql: str) -> tp.Sequence[tp.Dict[tp.Any, tp.Any]]:
    conn = Tortoise.get_connection("default")
    _, result = await conn.execute_query(sql)

    return result


async def truncate(model: models.Model):
    return await query(f'TRUNCATE TABLE {model._meta.db_table}')


async def get_size() -> str:
    result = await query(f"SELECT pg_database_size('{settings.DB_NAME}')/1024 AS kb_size;")

    size_mb = result[0].get('kb_size') / 1024  # type: ignore
    if size_mb // 1024 == 0:
        return f'{size_mb:.3f} MB'

    size_gb = size_mb / 1024
    return f'{size_gb:.3f} GB'
