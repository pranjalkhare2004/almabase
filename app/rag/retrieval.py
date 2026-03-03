"""pgvector-based retrieval with dynamic threshold filtering.

Responsibilities:
  - Index document chunks into PostgreSQL (persistent)
  - Retrieve + sort + filter chunks for a question via pgvector
"""
import logging
from typing import List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.rag.embedding import embed_texts, embed_query
from app.rag.chunking import chunk_document
from app.models import DocumentChunk

logger = logging.getLogger("rag.retrieval")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K = 5
ABSOLUTE_MIN_THRESHOLD = 0.30
DYNAMIC_MARGIN = 0.10


# ---------------------------------------------------------------------------
# Indexing (persistent via PostgreSQL)
# ---------------------------------------------------------------------------

def build_index(questionnaire_id: int, doc_text: str, doc_name: str, db: Session):
    """Chunk a document, embed, and store in PostgreSQL with pgvector."""
    chunks = chunk_document(doc_text, doc_name)
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    for i, chunk in enumerate(chunks):
        row = DocumentChunk(
            questionnaire_id=questionnaire_id,
            doc_name=chunk["doc_name"],
            chunk_id=i,
            chunk_text=chunk["text"],
            embedding=embeddings[i].tolist(),
        )
        db.add(row)

    db.commit()

    logger.info(
        "INDEXED | doc='%s' | chunks=%d | qid=%d",
        doc_name, len(chunks), questionnaire_id,
    )


# ---------------------------------------------------------------------------
# Retrieval + Filtering (pgvector cosine similarity)
# ---------------------------------------------------------------------------

def retrieve(
    questionnaire_id: int, question: str, db: Session
) -> Tuple[List[Tuple[float, dict]], List[Tuple[float, dict]], str]:
    """Retrieve, sort, and filter chunks using pgvector cosine similarity.

    Returns:
        all_retrieved: top_k chunks sorted by score desc
        selected: filtered + capped chunks ready for generation
        decision: "proceed" or "fallback"
    """
    query_vec = embed_query(question)
    vec_str = "[" + ",".join(str(float(v)) for v in query_vec[0]) + "]"

    result = db.execute(
        text("""
            SELECT
                id, questionnaire_id, doc_name, chunk_id, chunk_text,
                1 - (embedding <=> :qvec) AS similarity
            FROM document_chunks
            WHERE questionnaire_id = :qid
            ORDER BY embedding <=> :qvec
            LIMIT :topk
        """),
        {"qvec": vec_str, "qid": questionnaire_id, "topk": TOP_K},
    )
    rows = result.fetchall()

    if not rows:
        return [], [], "fallback"

    # Build sorted list
    retrieved: List[Tuple[float, dict]] = []
    for row in rows:
        retrieved.append((
            float(row.similarity),
            {
                "doc_name": row.doc_name,
                "chunk_id": row.chunk_id,
                "text": row.chunk_text,
            },
        ))
    retrieved.sort(key=lambda x: x[0], reverse=True)

    # Hard fallback gate
    best = retrieved[0][0]
    if best < ABSOLUTE_MIN_THRESHOLD:
        return retrieved, [], "fallback"

    # Dynamic margin filtering
    cutoff = best - DYNAMIC_MARGIN
    filtered = [(s, m) for s, m in retrieved if s >= cutoff]

    # Cap at 3
    selected = filtered[:3]

    if not selected:
        return retrieved, [], "fallback"

    return retrieved, selected, "proceed"
