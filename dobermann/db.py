import asyncio
import logging

import asyncpg

from . import config

logger = logging.getLogger(__name__)


conn: asyncpg.Connection = None

async def connect():
    conn = await asyncpg.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME,
        timeout=10,
    )
    return conn


async def close():
    await conn.close()


async def query():
    ...


def apply_migrations():
    ...


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['apply_migrations'])
    args = parser.parse_args()

    if args.command == 'apply_migrations':
        apply_migrations()


# TODO: https://github.com/MagicStack/asyncpg/issues/462#issuecomment-747053487