import asyncio
import datetime as dt
import functools
import logging
import signal
import typing as tp

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from dobermann.binance_client import client

from . import db
from .config import settings

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


def job(db: bool = True, loglevel: str = logging.INFO):
    def wrapper(func):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            job_id = func.__name__
            if args:
                job_id += f' {args}'
            if kwargs:
                job_id += f' {kwargs}'

            logger.log(loglevel, 'Job started: %s', job_id)
            res = await func(*args, **kwargs)

            logger.log(loglevel, 'Job finished: %s', job_id)
            return res

        return wrapped

    return wrapper


def add_job(
    task: tp.Awaitable,
    trigger: tp.Union[str, BaseTrigger, dt.timedelta],
    *,
    name: tp.Optional[str] = None,
) -> None:
    if isinstance(trigger, str):
        if trigger.isdigit():
            trigger = int(trigger)
        else:
            trigger = CronTrigger.from_crontab(trigger, timezone=settings.TIMEZONE)

    if isinstance(trigger, dt.timedelta):
        trigger = trigger.total_seconds()

    if isinstance(trigger, (int, float)):
        trigger = IntervalTrigger(seconds=trigger, timezone=settings.TIMEZONE)

    scheduler.add_job(func=task, trigger=trigger, name=name)


async def run_scheduler():
    loop = asyncio.get_event_loop()

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, stop_event.set)

    logger.info('Starting scheduler...')

    await db.init()
    await client.connect()
    scheduler.start()

    try:
        await stop_event.wait()

    finally:
        logger.info('Shutdown scheduler...')
        scheduler.shutdown()
        await db.close()
        await client.close()
