import asyncio
import datetime as dt
import decimal
import logging
import typing as tp
from decimal import Decimal

from binance import AsyncClient, BinanceSocketManager
from binance.enums import HistoricalKlinesType
from pydantic import BaseModel, Field, validator

from app.config import settings  # FIXME

from .utils import StrEnum

logger = logging.getLogger(__name__)


decimal.getcontext().rounding = decimal.ROUND_DOWN


class Model(BaseModel):

    class Config:
        orm_mode = True
        allow_mutation = False
        allow_population_by_field_name = True


class Timeframe(StrEnum):
    M1 = '1m'
    M3 = '3m'
    M5 = '5m'
    M15 = '15m'
    M30 = '30m'
    H1 = '1h'
    H2 = '2h'
    H4 = '4h'
    H6 = '6h'
    H8 = '8h'
    H12 = '12h'
    D1 = '1d'
    D3 = '3d'
    W1 = '1w'


class Asset(Model):
    ticker: str = Field(..., alias='symbol')
    status: str
    base_asset: str = Field(..., alias='baseAsset')
    quote_asset: str = Field(..., alias='quoteAsset')

    filters: tp.List[dict]


class FuturesAsset(Asset):
    underlying_type: str = Field(..., alias='underlyingType')
    onboard_date: int = Field(..., alias='onboardDate')

    @validator('onboard_date')
    def convert_ts(cls, value: tp.Union[dt.datetime, str]) -> dt.datetime:
        return dt.datetime.fromtimestamp(value / 1000).astimezone(settings.TIMEZONE)


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

    async def get_spot_assets(self) -> tp.List[Asset]:
        response = await self._client.get_exchange_info()
        return [
            Asset(**asset) for asset in response['symbols']
            if asset['quoteAsset'] == 'USDT'
            and asset['status'] == 'TRADING'
        ]

    async def get_futures_assets(self) -> tp.List[FuturesAsset]:
        response = await self._client.futures_exchange_info()
        return [
            FuturesAsset(**asset) for asset in response['symbols']
            if asset['quoteAsset'] == 'USDT'
            and asset['contractType'] == 'PERPETUAL'
        ]

    async def get_futures_historical_candles(
        self, ticker: str, timeframe: Timeframe, start: dt.datetime, end: dt.datetime
    ) -> tp.AsyncGenerator[None, Candle]:
        """Get historical klines for a futures

        Docs: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
        """
        start_ts = str(int(start.timestamp()))
        end_ts = str(int(end.timestamp()))

        data_generator = await self._client.futures_historical_klines_generator(
            symbol=ticker,
            interval=str(timeframe),
            start_str=start_ts,
            end_str=end_ts,
        )
        async for candle_data in data_generator:
            yield Candle(**dict(zip(self.CANDLE_HEADERS, candle_data)))

    async def get_recent_candles(self, ticker: str, timeframe: Timeframe, limit: int = 2) -> Candle:
        data = await self._client._klines(
            klines_type=HistoricalKlinesType.FUTURES,
            symbol=ticker,
            interval=timeframe,
            limit=limit,
        )
        return [Candle(**dict(zip(self.CANDLE_HEADERS, c))) for c in data]

    async def subscribe_futures_mark_price(
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

        timer: tp.Optional[float] = None

        async with self._ws_client.all_mark_price_socket(fast=True) as sock:
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


client = BinanceClient()
