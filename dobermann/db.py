from contextlib import asynccontextmanager
import typing as tp
from tortoise import Tortoise
from tortoise.connection import connections

from .config import settings

if tp.TYPE_CHECKING:
    from asyncpg import Connection
    from asyncpg.cursor import Cursor


async def connect() -> None:
    await Tortoise.init(settings.TORTOISE_ORM)


async def close() -> None:
    await Tortoise.close_connections()


@asynccontextmanager
async def connection() -> 'Connection':
    client = connections.get('default')

    async with client.acquire_connection() as conn:
        yield conn


@asynccontextmanager
async def cursor(sql: str, *args) -> 'Cursor':
    async with connection() as conn:
        async with conn.transaction():
            cursor = await conn.cursor(sql, *args)
            yield cursor


async def query(sql: str) -> tp.Sequence[dict]:
    conn = Tortoise.get_connection("default")
    _, result = await conn.execute_query(sql)

    return result


# TODO: https://github.com/MagicStack/asyncpg/issues/462#issuecomment-747053487
