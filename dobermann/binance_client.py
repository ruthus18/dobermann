import asyncio
import datetime as dt
import decimal
import enum
import logging
import typing as tp
from decimal import Decimal

from binance import AsyncClient, BinanceSocketManager
from pydantic import BaseModel, Field, validator

from app.config import settings  # FIXME

logger = logging.getLogger(__name__)


decimal.getcontext().rounding = decimal.ROUND_DOWN


class Model(BaseModel):

    class Config:
        orm_mode = True
        allow_mutation = False
        allow_population_by_field_name = True


class StrEnum(str, enum.Enum):

    def __str__(self) -> str:
        return self.value


class AccountType(StrEnum):
    SPOT = 'SPOT'
    MARGIN = 'MARGIN'


class BalanceAsset(Model):
    symbol: str = Field(alias='asset')
    amount_free: Decimal = Field(alias='free')
    amount_locked: Decimal = Field(alias='locked')
    account_type: AccountType

    @property
    def amount(self):
        return self.amount_free + self.amount_locked


class Timeframe(StrEnum):
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


class Candle(Model):
    open_time: dt.datetime
    open: Decimal
    close: Decimal
    low: Decimal
    high: Decimal
    volume: Decimal

    @validator('open_time')
    @classmethod
    def convert_tz(cls, value: tp.Union[dt.datetime, str]):
        return value.astimezone(settings.TIMEZONE)


class OrderSide(StrEnum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderType(StrEnum):
    # Suitable for futures exchange
    LIMIT = 'LIMIT'
    MARKET = 'MARKET'
    STOP = 'STOP'
    STOP_MARKET = 'STOP_MARKET'
    TAKE_PROFIT = 'TAKE_PROFIT'
    TAKE_PROFIT_MARKET = 'TAKE_PROFIT_MARKET'
    LIMIT_MAKER = 'LIMIT_MAKER'


class OrderTimeInForce(StrEnum):
    GTC = 'GTC'  # Good till cancelled
    IOC = 'IOC'  # Immediate or cancel
    FOK = 'FOK'  # Fill or kill


class BinanceClient:
    """Binance client wrapper

    Usage:
        A. Directly open and close:
            >>> client = BinanceClient(api_key, api_secret)
            >>> await client.connect()
            >>> ...
            >>> await client.close()

        B. Through async context manager:
            >>> async with BinanceClient(api_key, api_secret) as client:
            >>>     ...
    """

    CANDLE_HEADERS = (
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'buy_base_volume', 'buy_quote_volume', 'ignore'
    )

    def __init__(self, api_key: tp.Optional[str] = None, api_secret: tp.Optional[str] = None):
        self.__api_key = api_key or settings.BINANCE_KEY.get_secret_value()
        self.__api_secret = api_secret or settings.BINANCE_SECRET.get_secret_value()

        self._client: tp.Optional[AsyncClient] = None
        self._ws_client: tp.Optional[BinanceSocketManager] = None

        self._ws_subscriptions: tp.Dict[str, asyncio.Task] = {}

    async def connect(self):
        self._client = await AsyncClient.create(self.__api_key, self.__api_secret)
        self._ws_client = BinanceSocketManager(self._client)

    async def __aenter__(self):
        await self.connect()
        return self

    async def close(self):
        await self._client.close_connection()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

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

    # TODO: https://binance-docs.github.io/apidocs/futures/en/#futures-account-balance-v2-user_data
    async def get_balance_price(self) -> Decimal:
        total = Decimal(0)

        for asset in await self.get_balance_assets():
            if asset.symbol == 'USDT':
                total += asset.amount
                continue

            asset_price = await self._client.get_symbol_ticker(symbol=f'{asset.symbol}USDT')
            total += Decimal(asset_price['price']) * asset.amount

        return round(total, 2)

    # <DRAFT>
    # TODO: https://academy.binance.com/en/articles/understanding-the-different-order-types
    # async def create_futures_order(
    #     self,
    #     symbol: str,
    #     order_side: OrderSide,
    #     order_type: OrderType,
    #     quantity: Decimal,  # TODO: Cannot be sent with closePosition=true

    #     time_in_force: OrderTimeInForce = OrderTimeInForce.GTC,
    # ):
    #     """Create an order on futures exchange

    #     Docs: https://binance-docs.github.io/apidocs/futures/en/#new-order-trade
    #     """
    #     await self._client.futures_create_order(  # TODO
    #         symbol=symbol,
    #         side=order_side,
    #         type=order_type,
    #         quantity=quantity,
    #         timeInForce=time_in_force,
    #     )
    # </DRAFT>

    async def get_futures_historical_candles(
        self, symbol: str, timeframe: Timeframe, start: dt.datetime, end: dt.datetime
    ) -> tp.AsyncGenerator[None, Candle]:
        """Get historical klines for a futures

        Docs: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
        """
        start_ts = str(int(start.timestamp()))
        end_ts = str(int(end.timestamp()))

        data_generator = await self._client.futures_historical_klines_generator(
            symbol=symbol,
            interval=timeframe,
            start_str=start_ts,
            end_str=end_ts,
        )
        async for candle_data in data_generator:
            yield Candle(**dict(zip(self.CANDLE_HEADERS, candle_data)))

    async def subscribe_market_mark_price(
        self,
        callback: tp.Optional[tp.Awaitable] = None,
        rate_limit: tp.Optional[int] = None
    ):
        """Get latest prices for all futures assets on the market

        Docs: https://binance-docs.github.io/apidocs/futures/en/#mark-price-stream-for-all-market

        By default, socket receives new data every 2-3 seconds. `rate_limit` param is using to limit frequency
        of new data. For example, `rate_limit` = 60 means that callback will be reached every 60 seconds.
        """
        if callback is None:
            async def __callback(prices: dict):
                logger.info('Prices received, total assets: %s', len(prices))

            callback = __callback

        price_socket = self._ws_client.all_mark_price_socket(fast=False)
        timer: tp.Optional[float] = None

        async with price_socket as sock:
            logger.info('Start listening mark prices...')

            while True:
                message = await sock.recv()

                if rate_limit and rate_limit > 0:
                    current_ts = dt.datetime.fromtimestamp(message['data'][0]['E'] / 1000).replace(microsecond=0)

                    if timer is not None and (current_ts - timer).seconds < rate_limit:
                        continue

                    timer = current_ts

                await callback(message['data'])

    # <DRAFT>
    # async def subscribe_futures_candles(self, symbol: str, timeframe: Timeframe, callback: tp.Awaitable):
    #     trade_socket = self._ws_client.kline_futures_socket(symbol=symbol, interval=timeframe)

    #     last_ts_open = None
    #     async with trade_socket as sock:
    #         while True:
    #             res = await sock.recv()
    #             # TODO: Отдавать последнюю сформированную свечу, а не новую
    #             if last_ts_open != res['k']['t']:
    #                 await callback(res)

    #             last_ts_open = res['k']['t']

    # async def sub(self):
    #     import json

    #     redis = await aioredis.from_url('redis://127.0.0.1:6379', password='dobermann')

    #     async def callback(res: dict):
    #         await redis.publish('candles', json.dumps(res['k']))

    #     task = asyncio.create_task(self.subscribe_futures_candles('1000SHIBUSDT', '1m', callback))
    #     self._ws_subscriptions[id(task)] = task

    #     await asyncio.sleep(60)

    # </DRAFT>
