"""
Database Connection & Session Management
AI Pharmacovigilance Intelligence Platform

Provides both synchronous and asynchronous SQLAlchemy session factories
with proper connection pooling and health-check utilities.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from database.models import Base

# ---------------------------------------------------------------------------
# Resolve settings without circular imports
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from config.settings import settings
    _DB_URL: str = settings.database.DATABASE_URL
except Exception:
    _DB_URL = "sqlite:///./data/pharmacovigilance.db"

# Ensure data directory exists for SQLite
if _DB_URL.startswith("sqlite"):
    _db_path = _DB_URL.replace("sqlite:///", "").replace("sqlite://", "")
    Path(_db_path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sync engine & session
# ---------------------------------------------------------------------------

def _make_sync_url(url: str) -> str:
    """Convert async URL prefixes back to sync for the sync engine."""
    return url.replace("+aiosqlite", "").replace("+asyncpg", "")


_sync_url = _make_sync_url(_DB_URL)

engine = create_engine(
    _sync_url,
    connect_args={"check_same_thread": False} if "sqlite" in _sync_url else {},
    pool_pre_ping=True,
    echo=False,
)

# Enable WAL mode for SQLite — dramatically improves concurrent reads
if "sqlite" in _sync_url:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Async engine & session
# ---------------------------------------------------------------------------

def _make_async_url(url: str) -> str:
    """Ensure URL uses async driver prefix."""
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    return url


_async_url = _make_async_url(_DB_URL)

async_engine = create_async_engine(
    _async_url,
    connect_args={"check_same_thread": False} if "sqlite" in _async_url else {},
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession,
)


# ---------------------------------------------------------------------------
# Dependency injection helpers (FastAPI)
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """Synchronous DB session dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Asynchronous DB session dependency for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Context managers for service-layer use
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def db_session() -> Generator[Session, None, None]:
    """Synchronous session context manager for service-layer code."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextlib.asynccontextmanager
async def async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Asynchronous session context manager for service-layer code."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def create_all_tables() -> None:
    """Create all tables defined in the ORM models (idempotent)."""
    Base.metadata.create_all(bind=engine)


def drop_all_tables() -> None:
    """Drop all tables — use ONLY in development / testing."""
    Base.metadata.drop_all(bind=engine)


async def create_all_tables_async() -> None:
    """Async version of create_all_tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def health_check() -> dict:
    """Return DB connectivity health status."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": _sync_url}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc), "database": _sync_url}
