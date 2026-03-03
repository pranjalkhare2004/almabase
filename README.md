# Structured Questionnaire Answering Tool

**AI-powered tool that answers questionnaires using your reference documents with RAG (Retrieval-Augmented Generation).**

## Industry & Company

**Industry**: Compliance & Legal Tech  
**Company**: **AlmaBase Compliance Solutions** — A fictional startup that helps organizations automate regulatory questionnaire completion by matching questions to internal policy documents using AI.

## System Overview

This tool allows users to:
1. **Register/Login** with JWT-based authentication
2. **Upload a questionnaire** (PDF/TXT) — parsed into individual questions
3. **Upload reference documents** (PDF/TXT) — chunked and embedded into a FAISS vector store
4. **Generate answers** using RAG — each question retrieves relevant chunks, and Llama 3.3 (via Groq) generates grounded answers with citations
5. **Review & edit** answers before finalizing
6. **Export** the completed questionnaire as a formatted DOCX file

### Architecture

```
User → FastAPI (Jinja2 Templates) → SQLite (data) + FAISS (vectors)
                                   → Groq API (Llama 3.3 chat)
                                   → sentence-transformers (local embeddings)
                                   → python-docx (export)
```

## Assumptions

- Questions are separated by newlines or standard numbering (1. / 1) / Q1.)
- Reference documents contain the answers — the LLM only uses provided context
- FAISS index is in-memory (resets on server restart; suitable for demo/assignment use)
- `all-MiniLM-L6-v2` (sentence-transformers) for local embeddings, Groq `llama-3.3-70b-versatile` for generation
- Similarity threshold of 0.40 — below this, returns "Not found in references."

## Trade-offs

| Decision | Rationale |
|----------|-----------|
| In-memory FAISS | Simple, no external DB; acceptable for demo scope |
| SQLite | Zero config, file-based, perfect for single-instance deploy |
| Cookie-based JWT | Simpler than header-based auth for server-rendered templates |
| Minimal chunking (word-based) | Good enough for most documents; avoids tiktoken dependency |
| Local embeddings | No API cost for embeddings; sentence-transformers runs on CPU |
| No background workers | Answers generate synchronously; acceptable for small questionnaires |

## How to Run Locally

### Prerequisites
- Python 3.10+
- Groq API key (free at https://console.groq.com)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ALMABASE

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
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables: `GROQ_API_KEY`, `SECRET_KEY`

**Live URL**: _[To be added after deployment]_

## Project Structure

```
app/
  main.py          # FastAPI app, all routes
  auth.py          # Authentication (JWT, bcrypt)
  database.py      # SQLite + SQLAlchemy setup
  models.py        # Database models
  rag.py           # FAISS indexing, retrieval, answer generation
  utils.py         # Text extraction, question parsing, chunking
  templates/       # Jinja2 HTML templates
  static/          # CSS styles
requirements.txt
README.md
.env.example
```
