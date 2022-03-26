import logging.config

from app.config import settings

logging.config.dictConfig(settings.LOGGING)


DB_HOST = 'localhost'
DB_PORT = 5432
DB_USER = 'dobermann'
DB_PASSWORD = 'dobermann'
DB_NAME = 'dobermann'
DB_URI = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'