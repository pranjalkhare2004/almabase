"""FAISS-based retrieval with dynamic threshold filtering.

Responsibilities:
  - Manage per-questionnaire FAISS indexes (in-memory)
  - Index document chunks
  - Retrieve + sort + filter chunks for a question
"""
import logging
from typing import List, Dict, Tuple

import faiss

from app.rag.embedding import embed_texts, embed_query, EMBEDDING_DIM
from app.rag.chunking import chunk_document

logger = logging.getLogger("rag.retrieval")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K = 5
ABSOLUTE_MIN_THRESHOLD = 0.30  # Hard floor — below this, never proceed
DYNAMIC_MARGIN = 0.10          # Keep chunks within this margin of best score

# In-memory FAISS stores: {questionnaire_id: {"index": ..., "metadata": [...]}}
_stores: Dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(questionnaire_id: int, doc_text: str, doc_name: str):
    """Chunk a document and add to the FAISS index for a questionnaire."""
    chunks = chunk_document(doc_text, doc_name)
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    if questionnaire_id not in _stores:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        _stores[questionnaire_id] = {"index": index, "metadata": []}

    store = _stores[questionnaire_id]
    base = len(store["metadata"])
    store["index"].add(embeddings)

    for i, chunk in enumerate(chunks):
        store["metadata"].append({
            "doc_name": chunk["doc_name"],
            "chunk_id": base + i,
            "text": chunk["text"],
        })

    logger.info(
        "INDEXED | doc='%s' | chunks=%d | total=%d | qid=%d",
        doc_name, len(chunks), store["index"].ntotal, questionnaire_id,
    )


# ---------------------------------------------------------------------------
# Retrieval + Filtering
# ---------------------------------------------------------------------------

def retrieve(
    questionnaire_id: int, question: str
) -> Tuple[List[Tuple[float, dict]], List[Tuple[float, dict]], str]:
    """Retrieve, sort, and filter chunks for a question.

    Returns:
        all_retrieved: all top_k chunks sorted by score desc
        selected: filtered + capped chunks ready for generation
        decision: "proceed" or "fallback"
    """
    if questionnaire_id not in _stores:
        return [], [], "fallback"

    store = _stores[questionnaire_id]
    if store["index"].ntotal == 0:
        return [], [], "fallback"

    # Retrieve
    q_emb = embed_query(question)
    k = min(TOP_K, store["index"].ntotal)
    scores, indices = store["index"].search(q_emb, k)

    # Sort descending
    retrieved: List[Tuple[float, dict]] = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(store["metadata"]):
            retrieved.append((float(score), store["metadata"][idx]))
    retrieved.sort(key=lambda x: x[0], reverse=True)

    if not retrieved:
        return [], [], "fallback"

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
