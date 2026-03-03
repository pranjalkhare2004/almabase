"""RAG module: deterministic retrieval with dynamic threshold, system-controlled citations.

Architecture (per question):
  1. Embed question (local sentence-transformers)
  2. Retrieve top_k=5 chunks from FAISS
  3. Sort by similarity descending
  4. Hard fallback gate: if best score < ABSOLUTE_MIN_THRESHOLD → no LLM, no citations
  5. Dynamic filtering: keep chunks within MARGIN of best score (relative selection)
  6. Select top MAX_CONTEXT_CHUNKS from filtered set
  7. LLM generates answer text ONLY (unaware of chunk IDs / doc names)
  8. If LLM says "not found" → return fallback, no citations
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
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Config — tuned for balanced precision + recall
# ---------------------------------------------------------------------------
CHAT_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

ABSOLUTE_MIN_THRESHOLD = 0.30  # Hard floor — below this, never call LLM
DYNAMIC_MARGIN = 0.10          # Keep chunks within this margin of the best score
TOP_K_RETRIEVE = 5
MAX_CONTEXT_CHUNKS = 3

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
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # cosine sim on normalized vectors
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
        "INDEXED | doc='%s' | new_chunks=%d | total_vectors=%d | qid=%d",
        doc_name, len(chunks), store["index"].ntotal, questionnaire_id,
    )


# ---------------------------------------------------------------------------
# Retrieval: retrieve → sort → hard gate → dynamic filter → select
# ---------------------------------------------------------------------------

def _retrieve_sort_filter(
    questionnaire_id: int, question: str
) -> Tuple[
    List[Tuple[float, dict]],   # all_retrieved (sorted desc)
    List[Tuple[float, dict]],   # dynamically_filtered
    str,                        # decision: "fallback" | "proceed"
]:
    """Core retrieval logic with dynamic margin-based filtering.

    Returns (all_retrieved, filtered, decision).
    """
    if questionnaire_id not in faiss_stores:
        return [], [], "fallback"

    store = faiss_stores[questionnaire_id]
    if store["index"].ntotal == 0:
        return [], [], "fallback"

    # Step 1-2: Retrieve + sort
    query_emb = get_embeddings([question])
    k = min(TOP_K_RETRIEVE, store["index"].ntotal)
    scores, indices = store["index"].search(query_emb, k)

    retrieved: List[Tuple[float, dict]] = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(store["metadata"]):
            retrieved.append((float(score), store["metadata"][idx]))
    retrieved.sort(key=lambda x: x[0], reverse=True)

    if not retrieved:
        return [], [], "fallback"

    # Step 3: Hard fallback gate
    best_score = retrieved[0][0]
    if best_score < ABSOLUTE_MIN_THRESHOLD:
        return retrieved, [], "fallback"

    # Step 4: Dynamic margin-based filtering
    # Keep chunks within DYNAMIC_MARGIN of best_score
    cutoff = best_score - DYNAMIC_MARGIN
    filtered = [(s, m) for s, m in retrieved if s >= cutoff]

    if not filtered:
        return retrieved, [], "fallback"

    return retrieved, filtered, "proceed"


# ---------------------------------------------------------------------------
# LLM Generation (answer text ONLY — no citations)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise answering assistant. "
    "Answer ONLY using the provided context. "
    "Provide concise but complete information. "
    "Do NOT invent information beyond what the context states. "
    "Do NOT include citations, references, document names, or source identifiers. "
    "If the context does not contain the answer, respond exactly: "
    "'Not found in references.'"
)


def _generate_answer_text(context: str, question: str) -> Tuple[str, float]:
    """Call LLM to produce answer text only. Returns (answer, elapsed_seconds)."""
    prompt = f"""Answer the following question using ONLY the provided context.
Provide a concise but complete answer.
Do not invent information beyond what is stated in the context.
If the answer is not in the context, respond exactly: 'Not found in references.'
Do NOT include citations or source references.

Context:
{context}

Question: {question}

Answer:"""

    client = get_groq_client()
    start = time.time()
    resp = client.chat.completions.create(
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
    return resp.choices[0].message.content.strip(), elapsed


# ---------------------------------------------------------------------------
# Deterministic Citation Builder
# ---------------------------------------------------------------------------

def _build_citations(selected: List[Tuple[float, dict]]) -> str:
    """Build citation string from selected chunk metadata.
    Purely system-controlled — LLM never touches this.
    """
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
    q_short = question[:100]

    # ===================== RETRIEVAL =====================
    retrieved, filtered, decision = _retrieve_sort_filter(questionnaire_id, question)

    # ===================== OBSERVABILITY =====================
    logger.info("=" * 72)
    logger.info("QUESTION: %s", q_short)

    if not retrieved:
        logger.info("  ⚠ NO VECTORS in index → FALLBACK")
        return FALLBACK_ANSWER, ""

    best_score = retrieved[0][0]
    cutoff = best_score - DYNAMIC_MARGIN

    for rank, (score, meta) in enumerate(retrieved):
        passed = "✓" if score >= cutoff and best_score >= ABSOLUTE_MIN_THRESHOLD else "✗"
        logger.info(
            "  [%s] rank=%d  sim=%.4f  doc='%s'  chunk=%d  text='%s'",
            passed, rank, score, meta["doc_name"], meta["chunk_idx"],
            meta["text"][:70].replace("\n", " "),
        )

    logger.info(
        "  SCORES: best=%.4f  cutoff=%.4f (best - %.2f)  abs_min=%.4f",
        best_score, cutoff, DYNAMIC_MARGIN, ABSOLUTE_MIN_THRESHOLD,
    )

    # ===================== HARD FALLBACK GATE =====================
    if decision == "fallback":
        logger.info(
            "  ✗ FALLBACK | best=%.4f < abs_min=%.4f | NO LLM call | NO citations",
            best_score, ABSOLUTE_MIN_THRESHOLD,
        )
        return FALLBACK_ANSWER, ""

    # ===================== SELECT TOP CHUNKS =====================
    selected = filtered[:MAX_CONTEXT_CHUNKS]
    logger.info(
        "  ✓ SELECTED %d chunks | scores=%s",
        len(selected), [f"{s:.4f}" for s, _ in selected],
    )

    # ===================== GENERATION =====================
    context = "\n\n---\n\n".join(meta["text"] for _, meta in selected)
    answer, elapsed = _generate_answer_text(context, question)
    logger.info("  LLM: %.2fs | len=%d", elapsed, len(answer))

    # ===================== LLM FALLBACK RESPECT =====================
    if "not found in references" in answer.lower():
        logger.info("  LLM returned fallback → NO citations")
        return FALLBACK_ANSWER, ""

    # ===================== DETERMINISTIC CITATIONS =====================
    citations = _build_citations(selected)
    logger.info("  CITATIONS: %s", citations)
    logger.info("=" * 72)

    return answer, citations
