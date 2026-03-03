# Structured Questionnaire Answering Tool

**AI-powered tool that answers questionnaires using your reference documents with RAG (Retrieval-Augmented Generation).**

## Industry & Company

**Industry**: Compliance & Legal Tech  
**Company**: **AlmaBase Compliance Solutions** — A fictional startup that helps organizations automate regulatory questionnaire completion by matching questions to internal policy documents using AI.

## System Overview

This tool allows users to:
1. **Register/Login** with JWT-based authentication
2. **Upload a questionnaire** (PDF/TXT) — parsed into individual questions
3. **Upload reference documents** (PDF/TXT) — chunked and embedded into pgvector
4. **Generate answers** using RAG — each question retrieves relevant chunks, and Llama 3.3 (via Groq) generates grounded answers with citations
5. **Review & edit** answers before finalizing
6. **Export** the completed questionnaire as a formatted DOCX file

### Architecture

```
User → FastAPI (Jinja2 Templates) → PostgreSQL + pgvector (data + vectors)
                                   → Groq API (Llama 3.3 chat)
                                   → sentence-transformers (local embeddings)
                                   → python-docx (export)
```

### RAG Pipeline

```
Question → Embed → Retrieve top 5 (pgvector cosine) → Sort by similarity
  → Hard fallback gate (best < 0.30 → skip LLM)
  → Dynamic margin filter (best - 0.10)
  → Select top 3 chunks → LLM generates answer only
  → System attaches citations from chunk metadata
```

## Assumptions

- Questions are separated by newlines or standard numbering (1. / 1) / Q1.)
- Reference documents contain the answers — the LLM only uses provided context
- `all-MiniLM-L6-v2` (sentence-transformers) for local embeddings, Groq `llama-3.3-70b-versatile` for generation
- Dynamic threshold: absolute floor 0.30, margin 0.10 from best score

## Trade-offs

| Decision | Rationale |
|----------|-----------|
| PostgreSQL + pgvector | Persistent vectors that survive server restarts |
| Local embeddings | No API cost for embeddings; sentence-transformers runs on CPU |
| Cookie-based JWT | Simpler than header-based auth for server-rendered templates |
| Modular RAG (6 files) | Clean separation: chunking / embedding / retrieval / generation / citation / orchestrator |
| Dynamic margin threshold | Adapts to document quality; avoids rigid cutoff killing recall |

## How to Run Locally

### Prerequisites
- Python 3.10+
- Docker Desktop (for PostgreSQL)
- Groq API key (free at https://console.groq.com)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ALMABASE

# Start PostgreSQL with pgvector
docker-compose up -d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the application
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000` in your browser.

## Deployment

### Render

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repository
3. Add a PostgreSQL database (with pgvector extension)
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables: `GROQ_API_KEY`, `SECRET_KEY`, `DATABASE_URL`

**Live URL**: _[To be added after deployment]_

## Project Structure

```
app/
  main.py          # FastAPI app, all routes
  auth.py          # Authentication (JWT, bcrypt)
  database.py      # PostgreSQL + pgvector setup
  models.py        # Database models (incl. DocumentChunk with Vector)
  utils.py         # Text extraction, question parsing
  rag/
    __init__.py    # Package exports
    chunking.py    # Paragraph-aware chunking (400-700 tokens)
    embedding.py   # sentence-transformers (all-MiniLM-L6-v2)
    retrieval.py   # pgvector cosine similarity search
    generation.py  # Groq LLM (answer text only)
    citation.py    # Deterministic citation builder
    orchestrator.py # Pipeline orchestration
  templates/       # Jinja2 HTML templates
  static/          # CSS styles
docker-compose.yml
requirements.txt
README.md
.env.example
```

## What I'd Improve With More Time

- **Hybrid search**: Combine vector similarity with PostgreSQL full-text search (keyword boost)
- **Confidence scores**: Show similarity score alongside each answer
- **Evidence snippets**: Display the exact chunks used to generate each answer
- **Metadata enrichment**: Extract section headings and page numbers from documents
- **Version history**: Track answer edits over time
- **Batch processing**: Generate answers asynchronously for large questionnaires
