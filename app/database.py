"""PostgreSQL database setup using SQLAlchemy + pgvector."""
import os
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base


def _get_database_url() -> str:
    """Resolve DATABASE_URL strictly — no localhost fallback."""
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it in .env (local) or Render dashboard (production)."
        )

    # Render provides postgres:// but SQLAlchemy requires postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    return url


DATABASE_URL = _get_database_url()

# Log DB host at startup (no credentials) — uses print to guarantee visibility
_parsed = urlparse(DATABASE_URL)
print(f"[database] Connecting to DB host: {_parsed.hostname}:{_parsed.port}/{_parsed.path.lstrip('/')}")

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
