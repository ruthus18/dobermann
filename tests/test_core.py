import datetime as dt

import pytest

from dobermann import core


@pytest.mark.asyncio
async def test_candles_feed__get_all_tickers():
    start_at = dt.datetime(2020, 1, 1)
    end_at = dt.datetime(2020, 1, 2)
    timeframe = core.Timeframe.H1

    feed = core.CandlesFeed(start_at, end_at, timeframe, None)
    actual_tickers = await feed.get_actual_tickers()

    assert actual_tickers == {'BCHUSDT', 'BTCUSDT', 'ETHUSDT'}
    

@pytest.mark.asyncio
async def test_candles_feed__get_specific_tickers():
    start_at = dt.datetime(2022, 1, 1)
    end_at = dt.datetime(2022, 1, 2)
    timeframe = core.Timeframe.H1
    expected_tickers = {'LUNAUSDT', '1INCHUSDT'}

    feed = core.CandlesFeed(start_at, end_at, timeframe, expected_tickers)
    actual_tickers = await feed.get_actual_tickers()

    assert actual_tickers == expected_tickers
