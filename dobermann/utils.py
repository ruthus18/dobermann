import typing as tp
from decimal import Decimal

OptDecimal = tp.Optional[Decimal]


def RoundedDecimal(value: tp.Any) -> Decimal:
    return round(Decimal(value), 8)
