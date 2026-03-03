"""RAG module: local embeddings, FAISS indexing, retrieval, answer generation via Groq.

Pipeline (per question):
  1. Embed question (local sentence-transformers)
  2. Retrieve top_k=5 chunks from FAISS
  3. Filter chunks by similarity >= threshold
  4. If no chunk passes → return fallback, NO citations
  5. Select top 3 filtered chunks
  6. Generate answer using ONLY selected chunks (LLM produces answer text only)
  7. System attaches citations programmatically from chunk metadata
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
logging.basicConfig(level=logging.INFO)

groq_client: Optional[Groq] = None
embed_model: Optional[SentenceTransformer] = None

# In-memory FAISS store per questionnaire
faiss_stores: Dict[int, dict] = {}

CHAT_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.35
TOP_K_RETRIEVE = 5
MAX_CONTEXT_CHUNKS = 3
FALLBACK_ANSWER = "Not found in references."


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
    """Get embeddings using local sentence-transformers model."""
    model = get_embed_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def build_faiss_index(questionnaire_id: int, chunks: List[str], doc_name: str):
    """Add chunks to the FAISS index for a given questionnaire."""
    if not chunks:
        return

    embeddings = get_embeddings(chunks)

    if questionnaire_id not in faiss_stores:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
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
        "Indexed %d chunks from '%s' for questionnaire %d (total: %d)",
        len(chunks), doc_name, questionnaire_id, store["index"].ntotal,
    )


def retrieve_chunks(
    questionnaire_id: int, query: str, top_k: int = TOP_K_RETRIEVE
) -> List[Tuple[float, dict]]:
    """Retrieve top-k most similar chunks for a query."""
    if questionnaire_id not in faiss_stores:
        return []

    store = faiss_stores[questionnaire_id]
    if store["index"].ntotal == 0:
        return []

    query_embedding = get_embeddings([query])
    k = min(top_k, store["index"].ntotal)
    scores, indices = store["index"].search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(store["metadata"]):
            results.append((float(score), store["metadata"][idx]))
    return results


def filter_chunks_by_threshold(
    chunks: List[Tuple[float, dict]], threshold: float = SIMILARITY_THRESHOLD
) -> List[Tuple[float, dict]]:
    """Keep only chunks with similarity >= threshold, sorted by score desc."""
    filtered = [(score, meta) for score, meta in chunks if score >= threshold]
    filtered.sort(key=lambda x: x[0], reverse=True)
    return filtered


def build_citation_string(selected_chunks: List[Tuple[float, dict]]) -> str:
    """Build citation string programmatically from chunk metadata.
    LLM never touches this — it's system-controlled.
    """
    seen = []
    for _, meta in selected_chunks:
        citation = f"[{meta['doc_name']} - Chunk {meta['chunk_idx']}]"
        if citation not in seen:
            seen.append(citation)
    return ", ".join(seen)


def generate_answer(question: str, questionnaire_id: int) -> Tuple[str, str]:
    """Generate an answer for a question using the corrected RAG pipeline.

    Returns: (answer_text, citation_string)

    Pipeline:
      1. Retrieve top_k chunks
      2. Filter by similarity threshold
      3. If none pass → return fallback with NO citation
      4. Select top MAX_CONTEXT_CHUNKS
      5. LLM generates answer text only (no citations in prompt)
      6. System attaches citations from selected chunk metadata
    """
    # Step 1: Retrieve
    all_chunks = retrieve_chunks(questionnaire_id, question, top_k=TOP_K_RETRIEVE)

    # Log similarity scores
    logger.info("Question: %s", question[:80])
    for i, (score, meta) in enumerate(all_chunks):
        logger.info(
            "  Chunk %d: score=%.4f, doc='%s', chunk_idx=%d",
            i, score, meta["doc_name"], meta["chunk_idx"],
        )

    # Step 2: Filter by threshold
    filtered = filter_chunks_by_threshold(all_chunks, SIMILARITY_THRESHOLD)

    # Step 3: Fallback — no chunk passed threshold
    if not filtered:
        max_score = all_chunks[0][0] if all_chunks else 0.0
        logger.info(
            "  FALLBACK: max_score=%.4f < threshold=%.4f → skipping LLM",
            max_score, SIMILARITY_THRESHOLD,
        )
        return FALLBACK_ANSWER, ""

    # Step 4: Select top chunks for context (limit context overload)
    selected = filtered[:MAX_CONTEXT_CHUNKS]
    logger.info(
        "  Selected %d chunks (scores: %s)",
        len(selected), [f"{s:.4f}" for s, _ in selected],
    )

    # Step 5: Build context and call LLM (answer text ONLY — no citations)
    context = "\n\n---\n\n".join(meta["text"] for _, meta in selected)

    prompt = f"""Answer the following question using ONLY the provided context.
If the answer is not present in the context, respond exactly with:
'Not found in references.'

Do NOT include citations, source references, or document names in your answer.
Provide only the answer text.

Context:
{context}

Question: {question}

Answer:"""

    start_time = time.time()
    client = get_groq_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise answering assistant. "
                    "Answer ONLY from the given context. "
                    "Do NOT add citations or references. "
                    "If the answer is not in the context, say exactly: "
                    "'Not found in references.'"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    elapsed = time.time() - start_time
    logger.info("  LLM call took %.2fs", elapsed)

    answer = response.choices[0].message.content.strip()

    # Step 6: If LLM still says not found, respect it — no citations
    if answer == FALLBACK_ANSWER or "not found in references" in answer.lower():
        logger.info("  LLM returned fallback → no citations attached")
        return FALLBACK_ANSWER, ""

    # Step 7: System attaches citations programmatically
    citation = build_citation_string(selected)
    logger.info("  Citations attached: %s", citation)

    return answer, citation
