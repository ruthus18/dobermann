import asyncio
import itertools
import enum
import typing as tp
from decimal import Decimal

OptDecimal = tp.Optional[Decimal]


def RoundedDecimal(value: tp.Any) -> Decimal:
    return round(Decimal(value), 8)


class StrEnum(str, enum.Enum):

    def __str__(self) -> str:
        return self.value


async def cancel_task(task: asyncio.Task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def split_list_round_robin(data: list, chunks_num: int):
    """Divide list into n parts"""
    result = [[] for _ in range(chunks_num)]

    chunk_indexes = itertools.cycle(i for i in range(chunks_num))
    for item in data:
        i = next(chunk_indexes)
        result[i].append(item)

    return result
