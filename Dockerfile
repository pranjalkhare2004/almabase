FROM python:3.10-slim

WORKDIR /app

# System dependencies for psycopg2 and PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload and cache the embedding model during build (avoids cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Production settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Use $PORT from Render (falls back to 8000 for local Docker)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
