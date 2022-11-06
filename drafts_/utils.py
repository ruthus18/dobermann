import asyncio
import enum
import itertools
import time
import typing as tp
from collections import UserDict
from contextlib import contextmanager, suppress
from decimal import Decimal

import msgpack

if tp.TYPE_CHECKING:
    import asyncpg

OptDecimal = tp.Optional[Decimal]


def RoundedDecimal(value: tp.Any) -> Decimal:
    return round(Decimal(value), 8)


class StrEnum(str, enum.Enum):

    def __str__(self) -> str:
        return self.value


async def cancel_task(task: asyncio.Task):
    task.cancel()

    with suppress(asyncio.CancelledError):
        await task


def split_list_round_robin(data: tp.Iterable, chunks_num: int) -> tp.List[list]:
    """Divide iterable into `chunks_num` lists"""
    result = [[] for _ in range(chunks_num)]

    chunk_indexes = itertools.cycle(i for i in range(chunks_num))
    for item in data:
        i = next(chunk_indexes)
        result[i].append(item)

    return result


async def disable_decimal_conversion_codec(conn: 'asyncpg.Connection'):
    await conn.set_type_codec(
        'numeric',
        encoder=str,
        decoder=str,
        format='text',
        schema='pg_catalog',
    )


def packb_candle(candle: 'asyncpg.Record') -> bytes:
    return msgpack.packb({k: v for k, v in candle.items() if k != 'asset_id'}, datetime=True)


def unpackb_candle(data: bytes) -> dict:
    return msgpack.unpackb(data, timestamp=3)


class PerformanceStats(UserDict):
    
    @contextmanager
    def measure(self, name: str):
        start = time.time()
        yield
        self[name] = round(time.time() - start, 4)
