import asyncio
import datetime as dt
import functools
import logging
import typing as tp

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import models
from .config import settings

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


def job(db: bool = True, loglevel: str = logging.DEBUG):
    def wrapper(func):
        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            job_id = func.__name__
            if args:
                job_id += f' {args}'
            if kwargs:
                job_id += f' {kwargs}'

            logger.log(loglevel, 'Job started: %s', job_id)
            if db:
                await models.init_db()

            res = await func(*args, **kwargs)
            if db:
                await models.close_db()

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


def run_scheduler():
    loop = asyncio.get_event_loop()

    logger.info('Starting scheduler...')
    scheduler.start()

    try:
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info('Shutting down scheduler...')
    finally:
        # scheduler.shutdown(wait=False)
        loop.close()
