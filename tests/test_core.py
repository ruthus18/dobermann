import asyncio
import datetime as dt

import pytest
import zmq
from zmq.asyncio import Context, Poller

from dobermann import core
from dobermann.utils import unpackb_candle


@pytest.fixture(scope='module')
def zmq_ctx():
    return Context.instance()


@pytest.fixture
def expected_tickers():
    return {'LUNAUSDT', '1INCHUSDT', 'DYDXUSDT'}


@pytest.fixture
def feed(expected_tickers):
    start_at = dt.datetime(2022, 1, 1)
    end_at = dt.datetime(2022, 1, 2)
    timeframe = core.Timeframe.H1

    feed = core.CandlesFeed(start_at, end_at, timeframe, expected_tickers)
    yield feed
    feed.close()


@pytest.mark.asyncio
async def test_candles_feed__get_all_tickers():
    start_at = dt.datetime(2020, 1, 1)
    end_at = dt.datetime(2020, 1, 2)
    timeframe = core.Timeframe.H1

    feed = core.CandlesFeed(start_at, end_at, timeframe, None)
    actual_tickers = await feed.get_actual_tickers()

    assert actual_tickers == {'BCHUSDT', 'BTCUSDT', 'ETHUSDT'}
    feed.close()
    

@pytest.mark.asyncio
async def test_candles_feed__get_specific_tickers(feed, expected_tickers):
    actual_tickers = await feed.get_actual_tickers()

    assert actual_tickers == expected_tickers


@pytest.mark.asyncio
async def test_candles_feed__register_candle_recipients(zmq_ctx, feed, expected_tickers):
    registrator: zmq.Socket = zmq_ctx.socket(zmq.REP)
    registrator.bind(core.FEED_REGISTRY_URL)

    feed_registry_task = asyncio.create_task(feed.register_candle_recipients())

    data = await registrator.recv_multipart()
    await registrator.send(b'')

    assert set(data) == {t.encode() for t in expected_tickers}

    await asyncio.wait([feed_registry_task])
    registrator.close()


@pytest.mark.asyncio
async def test_candles_feed__send_candles(zmq_ctx, feed):
    candle_receiver = zmq_ctx.socket(zmq.SUB)
    candle_receiver.connect(core.CANDLES_URL)
    candle_receiver.subscribe(b'')

    candles_task = asyncio.create_task(feed.send_candles())

    candles = []
    while message := await candle_receiver.recv_multipart():
        match message:
            case [_, data]:  # ticker, data
                candles.append(data)

            case [core.EVENT_ALL_CANDLES_SENT]:
                break

    await asyncio.wait((candles_task, ))
    assert feed._total_candles_sent == len(candles)

    assert core.Candle(**unpackb_candle(candles[0]))

    candle_receiver.close()
