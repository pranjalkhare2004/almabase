"""PostgreSQL database setup using SQLAlchemy + pgvector."""
import os
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# --- Strict: crash if DATABASE_URL is not set ---
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Set it in .env (local) or Render dashboard (production)."
    )

# Render provides postgres:// but SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Print DB host for verification (no credentials)
_parsed = urlparse(DATABASE_URL)
print(f"[database] Connecting to DB host: {_parsed.hostname}")

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
        print("[database] pgvector extension enabled")
    except Exception as e:
        print(f"[database] WARNING: Could not enable pgvector extension: {e}")

    Base.metadata.create_all(bind=engine)
    print("[database] Tables created")
