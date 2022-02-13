import asyncio
import enum
import typing as tp
from contextlib import asynccontextmanager
from decimal import Decimal

OptDecimal = tp.Optional[Decimal]


def RoundedDecimal(value: tp.Any) -> Decimal:
    return round(Decimal(value), 8)


class StrEnum(str, enum.Enum):

    def __str__(self) -> str:
        return self.value


@asynccontextmanager
async def cancel_task(task: asyncio.Task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        yield
