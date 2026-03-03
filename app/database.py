"""PostgreSQL database setup using SQLAlchemy + pgvector."""
import os
import logging
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger("database")

# --- Strict DATABASE_URL resolution (no localhost fallback) ---
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Set it in .env (local) or Render dashboard (production)."
    )

# Render provides postgres:// but SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Log the DB host for verification (no credentials)
_parsed = urlparse(DATABASE_URL)
logger.info("Connecting to database host: %s:%s/%s", _parsed.hostname, _parsed.port, _parsed.path.lstrip("/"))

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Enable pgvector extension and create all tables."""
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        logger.info("pgvector extension enabled")
    except Exception as e:
        logger.warning("Could not enable pgvector extension: %s", e)

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
