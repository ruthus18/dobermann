import argparse
import logging.config

from .config import settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        choices=['sync'],
    )
    args = parser.parse_args()

    logging.config.dictConfig(settings.LOGGING)

    if args.command == 'sync':
        from . import market_sync
        market_sync.main()
