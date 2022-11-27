import asyncio
import datetime as dt
import itertools
import math

import httpx
from tqdm.asyncio import tqdm
from yarl import URL

from .config import logger
from .core import Asset, Candle, Timeframe

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
        if not response_data:
            self._handle_errors(response.json(), params)

        return response

    def _handle_errors(self, response_data: dict, params: dict | None = None) -> None:
        error_code = response_data['retCode']
        error_msg = response_data['retMsg']

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
        to_time: dt.datetime,
        batch_size: int = 200,
    ) -> list[Candle]:
        """Docs: https://bybit-exchange.github.io/docs/derivativesV3/unified_margin/#t-dv_querykline
        """
        response = await self.request(
            'GET',
            'derivatives/v3/public/kline',
            params={
                'category': 'linear',
                'symbol': asset,
                'interval': TIMEFRAME_MAP[timeframe],
                'start': int(from_time.timestamp() * 1000),
                # -1 becuase we need to exclude candle which open_time == to_time arg
                'end': int(to_time.timestamp() * 1000 - 1),
                'limit': batch_size,
            }
        )
        candles_data = response.json()['result']['list']
        return [
            Candle(
                open_time=dt.datetime.fromtimestamp(int(candle_data[0]) / 1000),
                open=float(candle_data[1]),
                high=float(candle_data[2]),
                low=float(candle_data[3]),
                close=float(candle_data[4]),
                volume=float(candle_data[5]),
            )
            for candle_data in candles_data
        ]

    async def _get_candles_slow(
        self,
        asset: Asset,
        timeframe: Timeframe,
        start_at: dt.datetime,
        end_at: dt.datetime,
    ) -> list[Candle]:
        MAX_BATCH_SIZE = 200

        pbar = tqdm(total=math.ceil((end_at - start_at) / timeframe.timedelta))
        start_at_ = start_at
        candles = []

        while start_at_ < end_at:
            response_candles = await self._get_candles_batch(
                asset, timeframe, start_at_, end_at, MAX_BATCH_SIZE
            )
            candles += response_candles

            start_at_ += timeframe.timedelta * MAX_BATCH_SIZE
            pbar.update(len(response_candles))

        pbar.close()
        return sorted(candles, key=lambda c: c['open_time'])

    async def _get_candles_fast(
        self,
        asset: Asset,
        timeframe: Timeframe,
        start_at: dt.datetime,
        end_at: dt.datetime,
    ) -> list[Candle]:
        MAX_BATCH_SIZE = 200
        MAX_TASKS = 45

        total_candles = math.ceil((end_at - start_at) / timeframe.timedelta)
        total_batches = math.ceil(total_candles / MAX_BATCH_SIZE)

        pbar = tqdm(total=total_candles)

        start_at_coll = [
            start_at + (timeframe.timedelta * MAX_BATCH_SIZE * i)
            for i in range(total_batches)
        ]
        limiter = asyncio.Semaphore(MAX_TASKS)

        async def fetch_task(start_at_: dt.datetime) -> list[Candle]:
            async with limiter:
                response_candles = await self._get_candles_batch(asset, timeframe, start_at_, end_at, MAX_BATCH_SIZE)
                await asyncio.sleep(1)

            pbar.update(len(response_candles))
            return response_candles

        tasks = [asyncio.create_task(fetch_task(start_at_)) for start_at_ in start_at_coll]
        responses = await asyncio.gather(*tasks)
        pbar.close()

        return sorted(list(itertools.chain.from_iterable(responses)), key=lambda c: c['open_time'])

    async def get_candles(
        self,
        asset: Asset,
        timeframe: Timeframe,
        start_at: dt.datetime,
        end_at: dt.datetime,
        *,
        concurrent: bool = True,
    ) -> list[Candle]:
        assert end_at > start_at
        logger.info('Downloading candles for <g>{}[{}]</g>...', asset, timeframe)

        # Test request
        response_candles = await self._get_candles_batch(asset, timeframe, start_at, end_at, batch_size=1)
        if len(response_candles) == 0:
            logger.warning('No candles found for {}[{}]', asset, timeframe)
            return []

        if response_candles[0]['open_time'] > start_at:
            start_at = response_candles[0]['open_time']

        if (now := dt.datetime.now()) < end_at:
            end_at = now

        if concurrent:
            coro = self._get_candles_fast
        else:
            coro = self._get_candles_slow

        return await coro(asset, timeframe, start_at, end_at)


client = BybitClient()
