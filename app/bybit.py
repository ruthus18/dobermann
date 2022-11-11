import asyncio
import datetime as dt
from functools import partial

import httpx
from tqdm.asyncio import tqdm
from yarl import URL

from .config import logger
from .core import Asset, Candle, Timeframe
from .utils import remove_candle_tiem_duplicates

TESTNET_URL = 'https://api-testnet.bybit.com'
MAINNET_URL = 'https://api.bybit.com'


TEST_ASSETS = ['BTCUSD', 'ETHUSD', 'BITUSD', 'XRPUSD', 'LTCUSD', 'SOLUSD']


TIMEFRAME_MAP = {
    Timeframe.M5: '5',
    Timeframe.H1: '60',
    Timeframe.H4: '240',
    Timeframe.D1: 'D',
}


class APIError(Exception):
    def __init__(self, code: str, message: str, *, request_params: dict | None = None):
        self.code = code
        self.messgae = message
        self.request_params = request_params or {}

    def __str__(self) -> str:
        s = f'[{self.code}] {self.messgae}'
        if self.request_params:
            s += f'; req_params={self.request_params}'

        return s


class BybitClient:
    def __init__(self, base_url: str = MAINNET_URL):
        self.base_url = URL(base_url)
        self._client = httpx.AsyncClient()

    async def connect(self) -> None:
        await self._client.__aenter__()

    async def __aenter__(self) -> 'BybitClient':
        await self.connect()
        return self

    async def close(self) -> None:
        await self._client.__aexit__()

    async def __aexit__(self, *_: list) -> None:
        await self.close()

    async def request(self, method: str, path: str, params: dict | None = None) -> httpx.Response:
        url = str(self.base_url / path)

        response = await self._client.request(method, url, params=params)
        response.raise_for_status()

        response_data = response.json()['result']
        if response_data is None:
            self._handle_errors(response.json(), params)

        return response

    def _handle_errors(self, response_data: dict, params: dict | None = None) -> None:
        error_code = response_data['ret_code']
        error_msg = response_data['ret_msg']

        raise APIError(error_code, error_msg, request_params=params or {})

    async def get_server_time(self) -> float:
        """Get actual server time (in timestamp format)

        Docs: https://bybit-exchange.github.io/docs/derivativesV3/unified_margin/#t-servertime
        """
        url = str(self.base_url / 'v2/public/time')

        response = await self._client.get(url)

        server_now = response.json()['time_now']
        return float(server_now)

    async def _get_candles_batch(
        self,
        asset: Asset,
        timeframe: Timeframe,
        from_time: dt.datetime,
        batch_size: int,
    ) -> list[Candle]:
        response = await self.request(
            'GET',
            'v2/public/kline/list',
            params={
                'category': 'linear',
                'symbol': asset,
                'interval': TIMEFRAME_MAP[timeframe],
                'from': int(from_time.timestamp()),
                'limit': batch_size,
            }
        )
        response_data = response.json()['result']
        if response_data is None:
            self._handle_errors(response.json())

        return sorted([
            Candle(
                open_time=dt.datetime.fromtimestamp(candle_data['open_time']),
                open=float(candle_data['open']),
                close=float(candle_data['close']),
                low=float(candle_data['low']),
                high=float(candle_data['high']),
                volume=float(candle_data['volume']),
            )
            for candle_data in response_data
        ], key=lambda c: c['open_time'])

    async def get_candles(  # noqa: C901
        self,
        asset: Asset,
        timeframe: Timeframe,
        start_at: dt.datetime,
        end_at: dt.datetime,
    ) -> list[Candle]:
        """Get historical candles which `open_time` within interval [start_at; end_at)

        Run `MAX_WORKERS` num of workers which make concurrent requests

        Docs: https://bybit-exchange.github.io/docs/derivativesV3/unified_margin/#t-dv_querykline
        """
        assert end_at > start_at
        logger.info('Downloading candles for <g>{}[{}]</g>...', asset, timeframe)

        _request = partial(self._get_candles_batch, asset, timeframe)

        # Test request
        response_data = await _request(start_at, batch_size=1)
        if len(response_data) == 0:
            # Cases when we call candles from future
            logger.warning('No candles found for {}[{}]', asset, timeframe)
            return []

        minimum_open_time = response_data[0]['open_time']
        if minimum_open_time >= end_at:
            # Cases when we call candles from past which is unknown
            logger.warning('No candles found for {}[{}]', asset, timeframe)
            return []

        # https://bybit-exchange.github.io/docs/futuresV2/inverse/#t-ipratelimits
        MAX_WORKERS = 12
        MAX_BATCH_SIZE = 200

        tasks_q: asyncio.Queue = asyncio.Queue()
        responses_data = []
        candles = []
        current_open_time = minimum_open_time

        pbar = tqdm()

        async def worker() -> None:
            while True:
                time = await tasks_q.get()
                response_data = await _request(time, MAX_BATCH_SIZE)
                responses_data.append(response_data)
                tasks_q.task_done()

        async def consumer() -> None:
            nonlocal responses_data
            nonlocal candles
            nonlocal current_open_time
            mark_exit = False

            while not mark_exit:
                time_batches = [
                    current_open_time + (timeframe.timedelta * MAX_BATCH_SIZE * i)
                    for i in range(MAX_WORKERS)
                ]
                for time in time_batches:
                    tasks_q.put_nowait(time)

                await tasks_q.join()

                current_candles = []
                for response_data in responses_data:
                    if len(response_data) == 0:
                        mark_exit = True

                    current_candles += response_data

                if len(current_candles) == 0:
                    break

                current_candles.sort(key=lambda c: c['open_time'])
                max_open_time = current_candles[-1]['open_time']

                if max_open_time > end_at:
                    mark_exit = True
                    current_candles = [c for c in current_candles if c['open_time'] < end_at]

                candles += current_candles
                pbar.update(len(current_candles))

                responses_data = []
                current_open_time = max_open_time + timeframe.timedelta

                if current_open_time >= end_at:
                    mark_exit = True

        workers = [asyncio.create_task(worker()) for _ in range(MAX_WORKERS)]
        consumer_task = asyncio.create_task(consumer())

        await asyncio.wait([consumer_task])
        for worker in workers:
            worker.cancel()

        pbar.close()
        remove_candle_tiem_duplicates(candles)

        logger.success('Downloaded {} candles for {}[{}]', len(candles), asset, timeframe)
        return candles


client = BybitClient()
