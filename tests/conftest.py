import asyncio
import typing as tp

import pytest

from dobermann import db

if tp.TYPE_CHECKING:
    from asyncpg.transaction import Transaction



@pytest.fixture(scope="session")
def event_loop():
    return asyncio.get_event_loop()


@pytest.fixture(autouse=True, scope='session')
async def database() -> tp.AsyncGenerator:
    await db.connect()
    yield
    await db.close()
