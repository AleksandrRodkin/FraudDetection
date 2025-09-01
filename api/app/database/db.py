"""
Connect to DB
"""
from urllib.parse import urlparse

import asyncpg
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from api.logger_config import log
# Load keys
from ..config.conf import DB_HOST, DB_PORT, DB_NAME, DB_PASSWORD, DB_USER

DATABASE_URL = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Path to DB
engine = create_async_engine(url=DATABASE_URL)

# Connection to DB
async_session = async_sessionmaker(engine)


class Base(AsyncAttrs, DeclarativeBase):
    pass


async def ensure_database_exists(db=DATABASE_URL):
    """Check if the database exists and create it if not"""

    parsed_url = urlparse(db)

    db_name = parsed_url.path.lstrip("/")
    user = parsed_url.username
    password = parsed_url.password
    host = parsed_url.hostname
    port = parsed_url.port or 5432

    # Connection to postgres
    system_db_url = f"postgresql://{user}:{password}@{host}:{port}/postgres"

    conn = await asyncpg.connect(system_db_url)
    try:
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            db_name
        )
        if not db_exists:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            log.info("Database created")

        else:
            tables = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public')"
            )
            if tables:
                return True
            else:
                return False

    finally:
        await conn.close()


async def init_db():
    """Initializes the database by creating all tables specified in the Base metadata."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        log.info("Connection to the database established")
