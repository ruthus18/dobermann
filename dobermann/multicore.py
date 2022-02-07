import asyncio
import logging.config
import time
import typing as tp
from multiprocessing import Process
from unittest import result

import zmq
from zmq.asyncio import Context

from app.config import settings

from .base import Strategy
from .binance_client import BinanceClient

logging.config.dictConfig(settings.LOGGING)
logger = logging.getLogger(__name__)


# def worker():
#     # init binance client

#     while event := socket.wait_event():  # <-- timeout
#         if event.name == 'start_backtest':
#             signals = backtest(**event.params)  # <-- socket.send_event('tick_backtest')
#             socket.send_event('done_backtest', signals)

#         elif event.name == 'chill':
#             return


# def manager(tickers, timeframes, start_at, end_at):
#     for ticker in tickers:
#         for timeframe in timeframes:
#             socket.send_event('start_backtest', ticker, timeframe, start_at, end_at)

#     for _ in range(POOL_SIZE):
#         # После того, как воркер выполнил последнюю задачу - он может не ждать остальных
#         # Поэтому заранее кладем событие об отдыхе в конец очереди
#         socket.send_event('chill')

#     done = 0
#     total = len(tickers) * len(timeframes) * len(total_ticks)
#     signals = []

#     while done < total:  # <-- timeout
#         event = socket.wait_event()

#         if event.name == 'tick_backtest':
#             done += 1

#         elif event.name == 'done_backtest':
#             signals.append(event.params)

#     return signals


POOL_SIZE = 6
JOB_URL = 'tcp://127.0.0.1:5555'
RESULT_URL = 'tcp://127.0.0.1:5556'


class Worker:

    def __init__(self, worker_id: int):
        self.worker_id = worker_id

        self.client = BinanceClient()

        self.zmq_context = Context.instance()
        self.job_socket = self.zmq_context.socket(zmq.PULL)
        self.job_socket.connect(JOB_URL)

        self.result_socket = self.zmq_context.socket(zmq.PUSH)
        self.result_socket.connect(RESULT_URL)

    @classmethod
    def spawn(cls, *args):
        self = cls(*args)
        loop = asyncio.get_event_loop()

        loop.run_until_complete(self.run())
        loop.run_until_complete(self.stop())

    async def run(self):
        await self.client.connect()

        # Send ready signal to manager
        await self.result_socket.send(b'0')

        while True:
            ...

    async def stop(self):
        await self.client.close()


async def main():
    ctx = Context.instance()

    logger.info('Starting ZMQ sockets')

    job_socket = ctx.socket(zmq.PUSH)
    job_socket.bind(JOB_URL)

    result_socket = ctx.socket(zmq.PULL)
    result_socket.bind(RESULT_URL)

    logger.info('Starting workers: n=%s', POOL_SIZE)

    for i in range(POOL_SIZE):
        worker_id = i + 1
        process = Process(target=Worker.spawn, args=(worker_id,))
        process.start()

    for i in range(POOL_SIZE):
        await result_socket.recv()

    logger.info('All workers started, processing tasks...')

    logger.info('Waiting workers to finish...')

    # is_running = False
    # while is_running:
    #     message = await result_socket.recv()


    logger.info('Complete!')


if __name__ == '__main__':
    asyncio.run(main())
