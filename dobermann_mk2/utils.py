from .core import Candle


def check_candle_time_duplicates(candles: list[Candle]) -> list[int]:
    """Check duplicated `open_time` param in candle list.

    Should be executed on sorted list of candles
    """
    items_to_pop = []
    for i in range(len(candles) - 1):
        if candles[i]['open_time'] == candles[i+1]['open_time']:
            items_to_pop.append(i)

    return items_to_pop


def remove_candle_tiem_duplicates(candles: list[Candle]) -> None:
    items_to_pop = check_candle_time_duplicates(candles)

    offset = 0
    for idx in items_to_pop:
        candles.pop(idx - offset)
        offset += 1
