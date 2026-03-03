"""RAG module: deterministic retrieval, system-controlled citations, hard fallback.

Architecture (per question):
  1. Embed question (local sentence-transformers)
  2. Retrieve top_k=5 chunks from FAISS
  3. Sort by similarity descending
  4. Hard fallback gate: if best score < threshold → return fallback, NO LLM, NO citations
  5. Filter: keep only chunks with score >= threshold
  6. Select top 3 filtered chunks
  7. LLM generates answer text ONLY (no citations in prompt, LLM unaware of chunk IDs)
  8. If LLM returns fallback text → return fallback, NO citations
  9. System attaches citations deterministically from selected chunk metadata
"""
import os
import time
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHAT_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

SIMILARITY_THRESHOLD = 0.45   # Hard gate — nothing below this touches LLM or citations
TOP_K_RETRIEVE = 5            # Retrieve 5, filter+select from there
MAX_CONTEXT_CHUNKS = 3        # Max chunks sent to LLM (prevents context blending)

FALLBACK_ANSWER = "Not found in references."

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
groq_client: Optional[Groq] = None
embed_model: Optional[SentenceTransformer] = None
faiss_stores: Dict[int, dict] = {}


def get_groq_client() -> Groq:
    global groq_client
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        groq_client = Groq(api_key=api_key)
    return groq_client


def get_embed_model() -> SentenceTransformer:
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Embed texts locally. Returns L2-normalized float32 array."""
    model = get_embed_model()
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")


# ---------------------------------------------------------------------------
# FAISS Index Management
# ---------------------------------------------------------------------------

def build_faiss_index(questionnaire_id: int, chunks: List[str], doc_name: str):
    """Add document chunks to the FAISS index for a questionnaire."""
    if not chunks:
        return

    embeddings = get_embeddings(chunks)

    if questionnaire_id not in faiss_stores:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product on normalized = cosine
        faiss_stores[questionnaire_id] = {"index": index, "metadata": []}

    store = faiss_stores[questionnaire_id]
    base_idx = len(store["metadata"])
    store["index"].add(embeddings)

    for i, chunk in enumerate(chunks):
        store["metadata"].append({
            "doc_name": doc_name,
            "chunk_idx": base_idx + i,
            "text": chunk,
        })

    logger.info(
        "INDEXED | doc='%s' | chunks=%d | total_vectors=%d | qid=%d",
        doc_name, len(chunks), store["index"].ntotal, questionnaire_id,
    )


# ---------------------------------------------------------------------------
# Step 1-3: Retrieve + Sort + Filter
# ---------------------------------------------------------------------------

def _retrieve_and_filter(
    questionnaire_id: int, question: str
) -> Tuple[List[Tuple[float, dict]], List[Tuple[float, dict]]]:
    """Retrieve top_k chunks, return (all_retrieved, filtered_by_threshold).

    Both lists are sorted by similarity descending.
    """
    if questionnaire_id not in faiss_stores:
        return [], []

    store = faiss_stores[questionnaire_id]
    if store["index"].ntotal == 0:
        return [], []

    query_emb = get_embeddings([question])
    k = min(TOP_K_RETRIEVE, store["index"].ntotal)
    scores, indices = store["index"].search(query_emb, k)

    # Build sorted list (already sorted by FAISS, but be explicit)
    retrieved: List[Tuple[float, dict]] = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(store["metadata"]):
            retrieved.append((float(score), store["metadata"][idx]))
    retrieved.sort(key=lambda x: x[0], reverse=True)

    # Filter by threshold
    filtered = [(s, m) for s, m in retrieved if s >= SIMILARITY_THRESHOLD]

    return retrieved, filtered


# ---------------------------------------------------------------------------
# Step 7-8: LLM Generation (answer text only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise answering assistant. "
    "Answer ONLY using the provided context. "
    "Do NOT add citations, references, document names, or chunk IDs. "
    "If the context does not contain the answer, respond exactly: "
    "'Not found in references.'"
)


def _generate_answer_text(context: str, question: str) -> Tuple[str, float]:
    """Call LLM to generate answer text only. Returns (answer, elapsed_seconds)."""
    prompt = f"""Answer the following question using ONLY the provided context.
If the answer is not in the context, respond exactly: 'Not found in references.'

Do NOT include citations, source names, or document references.

Context:
{context}

Question: {question}

Answer:"""

    client = get_groq_client()
    start = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=500,
    )
    elapsed = time.time() - start
    return response.choices[0].message.content.strip(), elapsed


# ---------------------------------------------------------------------------
# Step 9: Deterministic Citation Attachment
# ---------------------------------------------------------------------------

def _build_citations(selected: List[Tuple[float, dict]]) -> str:
    """Build citation string from chunk metadata. System-controlled, never LLM."""
    seen: List[str] = []
    for _, meta in selected:
        c = f"[{meta['doc_name']} - Chunk {meta['chunk_idx']}]"
        if c not in seen:
            seen.append(c)
    return ", ".join(seen)


# ---------------------------------------------------------------------------
# Public API: generate_answer
# ---------------------------------------------------------------------------

def generate_answer(question: str, questionnaire_id: int) -> Tuple[str, str]:
    """Full deterministic RAG pipeline for one question.

    Returns: (answer_text, citation_string)
    """
    q_preview = question[:100]

    # --- Retrieve + Filter ---
    retrieved, filtered = _retrieve_and_filter(questionnaire_id, question)

    # --- Observability: log all retrieved scores ---
    logger.info("=" * 70)
    logger.info("QUESTION: %s", q_preview)
    if not retrieved:
        logger.info("  NO VECTORS in index → fallback")
        return FALLBACK_ANSWER, ""

    for i, (score, meta) in enumerate(retrieved):
        marker = "✓" if score >= SIMILARITY_THRESHOLD else "✗"
        logger.info(
            "  [%s] rank=%d  score=%.4f  doc='%s'  chunk=%d  preview='%s'",
            marker, i, score, meta["doc_name"], meta["chunk_idx"],
            meta["text"][:60].replace("\n", " "),
        )

    # --- Hard Fallback Gate (Phase 3) ---
    best_score = retrieved[0][0]
    if not filtered:
        logger.info(
            "  FALLBACK TRIGGERED | best_score=%.4f < threshold=%.4f | NO LLM call",
            best_score, SIMILARITY_THRESHOLD,
        )
        return FALLBACK_ANSWER, ""

    # --- Select top N filtered chunks (Phase 4) ---
    selected = filtered[:MAX_CONTEXT_CHUNKS]
    logger.info(
        "  SELECTED %d chunks | scores=%s",
        len(selected), [f"{s:.4f}" for s, _ in selected],
    )

    # --- Build context + Generate (Phase 2, 6) ---
    context = "\n\n---\n\n".join(meta["text"] for _, meta in selected)
    answer, elapsed = _generate_answer_text(context, question)
    logger.info("  LLM responded in %.2fs | answer_len=%d", elapsed, len(answer))

    # --- If LLM says not found → respect it, no citations (Phase 3 secondary) ---
    if "not found in references" in answer.lower():
        logger.info("  LLM returned fallback → NO citations attached")
        return FALLBACK_ANSWER, ""

    # --- Attach citations deterministically (Phase 5) ---
    citations = _build_citations(selected)
    logger.info("  CITATIONS: %s", citations)

    return answer, citations
