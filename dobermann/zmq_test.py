import asyncio
import time

import zmq
from zmq.asyncio import Context, Poller


# 1. Asyncio start

# url = 'tcp://127.0.0.1:4444'
# ctx = Context.instance()

# async def sender():
#     socket = ctx.socket(zmq.PUSH)
#     socket.bind(url)

#     while True:
#         print('Send Event')
#         await socket.send_string(f'Hello World! {time.time()}')
#         await asyncio.sleep(3)


# async def receiver():
#     socket = ctx.socket(zmq.PULL)
#     socket.connect(url)

#     while True:
#         message = await socket.recv()
#         print(f'Received event: {message}')


# if __name__ == '__main__':
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(
#         asyncio.wait([
#             sender(),
#             receiver(),
#         ])
#     )


# 2. Request-Reply

url = 'tcp://127.0.0.1:5555'
ctx = Context.instance()


async def client():
    socket = ctx.socket(zmq.REQ)
    socket.connect(url)

    for request in range(10):
        print(f'Send request {request} ...')
        await socket.send_string(f'Hello')

    await socket.recv()


async def server():
    socket = ctx.socket(zmq.PEP)
    socket.bind(url)

    for request in range(10):
        message = await socket.recv()
        print(f'Received event: {message}')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        asyncio.wait([
            sender(),
            receiver(),
        ])
    )
