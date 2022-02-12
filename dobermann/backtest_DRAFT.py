from abc import ABC, abstractmethod
import asyncio
from itertools import cycle
import typing as tp

from .base import Candle, Ticker, Timeframe

import zmq
from zmq.asyncio import Context


socket = None

POOL_SIZE = 6


class Strategy(ABC):
    """Базовый класс для реализации торговых стратегий.
    
    Стратегия - это сущность, которая принимает на вход данные о свечах и дает сигналы о том, когда покупать/продавать
    тот или иной актив. Данные о свечах передаются в индикаторы, на основе которых стратегия принимает решения.

    [DEV NOTES]
    Дальнейшие улучшения:
        * Для лучшей масштабируемости стоит рассмотреть вынесение обработки индикаторов за границы стратегии.
          Это нужно для ситуаций, когда несколько стратегий используют одни и те же индикаторы
    """
    def __init__(self): ...

    @abstractmethod
    def on_candle(self, candle: Candle) -> None: ...


class DataFeed:

    def run(self):
        ...
        socket.send_new_candle('BTCUSDT', {'o': ..., 'h': ...})


class Worker:
    ...


class Exchange:
    ...


# TODO: Сейчас допущение, что timeframe и стратегия всего одна.
#       Можно обобщить до произвольного количества стратегий и таймфреймов
class Manager:

    def __init__(self, strategy_cls: tp.Type[Strategy], tickers: tp.List[Ticker]):
        self.strategy_cls = strategy_cls
        self.tickers = tickers

        self.asset_router = {}  # ticker -> worker_id
        self.worker_ids = []
        
        zmq_ctx = Context.instance()
        self.feed_sock_in = zmq_ctx.socket(zmq.SUB)    # Входящий поток свечей от DataFeed
        self.logs_sock_in = zmq_ctx.socket(zmq.PULL)    # Входящий поток событий от воркеров (для целей логирования)
        self.events_sock_out = zmq_ctx.socket(zmq.PUB)  # Исходящий поток свечей и пр событий для воркеров

    def launch_workers(self):
        # Запускаем воркеров и заполняем self.worker_ids
        ...

    async def register_asset(self, ticker: Ticker, worker_id: str):
        msg = f'{worker_id};register;{ticker}'

        await self.events_sock_out.send_string()
        await self.logs_sock_in.recv_string()
        self.asset_router[ticker] = worker_id

    async def setup(self):
        self.events_sock_out.bind(...)
        self.logs_sock_in.bind(...)

        self.feed_sock_in.connect(...)

        self.launch_workers()

        workers_iter = cycle(self.worker_ids)
        for ticker in self.tickers:
            await self.register_asset(ticker, next(workers_iter))

    async def run_feed_listener(self):
        while True:
            candle = await self.feed_sock_in.recv_string()

            ...

    async def run_logs_listener(self):
        while True:
            event = await self.events_sock_out.recv_string()

            ...

    async def shutdown(self):
        for worker_id in self.worker_ids:
            await self.events_sock_out.send_string(f'{worker_id};chill;')

        self.feed_sock_in.close()
        self.events_sock_out.close()
        self.logs_sock_in.close()

    as
