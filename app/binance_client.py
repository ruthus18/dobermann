import datetime as dt
import decimal
import enum
import typing as tp
from decimal import Decimal

import pandas as pd
from binance.client import AsyncClient
from pydantic import BaseModel, Field, validator
from tqdm.asyncio import tqdm_asyncio

from .config import settings

decimal.getcontext().rounding = decimal.ROUND_DOWN


class _Model(BaseModel):
    class Config:
        orm_mode = True
        allow_mutation = False
        allow_population_by_field_name = True


class AccountType(str, enum.Enum):
    SPOT = 'SPOT'
    MARGIN = 'MARGIN'

    def __str__(self) -> str:
        return self.value


class BalanceAsset(_Model):
    symbol: str = Field(alias='asset')
    amount_free: Decimal = Field(alias='free')
    amount_locked: Decimal = Field(alias='locked')
    account_type: AccountType

    @property
    def amount(self):
        return self.amount_free + self.amount_locked


class CandleInterval(str, enum.Enum):
    M1 = '1m'
    M3 = '3m'
    M5 = '5m'
    M15 = '15m'
    M30 = '30m'
    H1 = 'h1'
    H2 = 'h2'
    H4 = 'h4'
    H6 = 'h6'
    H8 = 'h8'
    H12 = 'h12'
    D1 = 'd1'
    D3 = 'd3'
    W1 = 'w1'


class Candle(_Model):
    open_time: dt.datetime
    close_time: dt.datetime
    open: Decimal
    close: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal

    @classmethod
    @validator('open_time', 'close_time', pre=True)
    def convert_time(cls, value: tp.Union[dt.datetime, str]):
        if isinstance(value, str):
            return dt.datetime.fromtimestamp(value, tz=settings.TIMEZONE)

        return value


class BinanceClient:
    CANDLE_HEADERS = (
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'buy_base_volume', 'buy_quote_volume', 'ignore'
    )

    def __init__(self, client: AsyncClient):
        self._client = client

    @classmethod
    async def init(cls, api_key: str = None, api_secret: str = None):
        api_key = api_key or settings.BINANCE_KEY.get_secret_value()
        api_secret = api_secret or settings.BINANCE_SECRET.get_secret_value()

        client = await AsyncClient.create(api_key, api_secret)
        return cls(client)

    async def close(self):
        await self._client.close_connection()

    async def _get_spot_balance_assets(self) -> tp.List[BalanceAsset]:
        spot_account_data = await self._client.get_account()
        return [
            BalanceAsset(**item, account_type=AccountType.SPOT)
            for item in spot_account_data['balances'] if float(item['free']) != 0
        ]

    async def _get_margin_balance_assets(self) -> tp.List[BalanceAsset]:
        margin_assets = []
        margin_account_data = await self._client.get_isolated_margin_account()

        for base_item in margin_account_data['assets']:
            for item_type in ('baseAsset', 'quoteAsset'):
                item = base_item[item_type]

                margin_assets.append(
                    BalanceAsset(
                        symbol=item['asset'],
                        amount_free=Decimal(item['free']) - Decimal(item['borrowed']),
                        amount_locked=item['locked'],
                        account_type=AccountType.MARGIN
                    )
                )

        return margin_assets

    # TODO
    async def _get_futures_balance_assets(self) -> tp.List[BalanceAsset]:
        futures_assets = await self._client.get_subaccount_futures_details()
        return futures_assets

    async def get_balance_assets(self) -> tp.List[BalanceAsset]:
        spot_assets = await self._get_spot_balance_assets()
        margin_assets = await self._get_margin_balance_assets()
        futures_assets = await self._get_futures_balance_assets()

        return spot_assets + margin_assets + futures_assets

    async def get_balance_price(self) -> Decimal:
        total = Decimal(0)

        for asset in await self.get_balance_assets():
            if asset.symbol == 'USDT':
                total += asset.amount
                continue

            asset_price = await self._client.get_symbol_ticker(symbol=f'{asset.symbol}USDT')
            total += Decimal(asset_price['price']) * asset.amount

        return round(total, 2)

    async def get_futures_historical_candles(
        self, symbol: str, interval: CandleInterval, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        start_ts = str(int(start.timestamp()))
        end_ts = str(int(end.timestamp()))

        data_generator = tqdm_asyncio(
            await self._client.futures_historical_klines_generator(
                symbol=symbol,
                interval=interval,
                start_str=start_ts,
                end_str=end_ts,
            )
        )
        candles: tp.Generator[Candle] = [
            Candle(**dict(zip(self.CANDLE_HEADERS, candle_data)))
            async for candle_data in data_generator
        ]

        return pd.DataFrame(c.__dict__ for c in candles).sort_values('open_time')
