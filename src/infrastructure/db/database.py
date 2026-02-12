"""
Database connection management.

Supports:
  - SQLite (local dev, no setup)
  - PostgreSQL + pgvector (Docker / GCP Cloud SQL)

Connection string comes from DATABASE_URL env var.
"""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.infrastructure.db.models import Base

logger = logging.getLogger(__name__)

# Default to SQLite for zero-setup local dev
DEFAULT_DB_URL = "sqlite:///fraud_doc.db"


def get_database_url() -> str:
    """Get database URL from environment."""
    return os.getenv("DATABASE_URL", DEFAULT_DB_URL)


def create_db_engine(url: str = None):
    """Create SQLAlchemy engine."""
    db_url = url or get_database_url()

    if db_url.startswith("sqlite"):
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    else:
        # PostgreSQL
        engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )

    return engine


# ── Global engine & session factory ──
_engine = None
_SessionFactory = None


def get_engine():
    """Get or create the global engine."""
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


def init_db():
    """Create all tables. Safe to call multiple times."""
    engine = get_engine()
    db_url = get_database_url()

    # Enable pgvector extension if PostgreSQL
    if "postgresql" in db_url or "postgres" in db_url:
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable pgvector: {e}")

    Base.metadata.create_all(engine)
    logger.info(f"Database initialized: {db_url.split('@')[-1] if '@' in db_url else db_url}")


@contextmanager
def get_db() -> Session:
    """Context manager for database sessions."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
