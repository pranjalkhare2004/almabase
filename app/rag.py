"""RAG module: embeddings, FAISS indexing, retrieval, answer generation."""
import os
from typing import List, Dict, Tuple, Optional
import numpy as np

import faiss
from openai import OpenAI

client: Optional[OpenAI] = None
# In-memory FAISS store per questionnaire
# Structure: {questionnaire_id: {"index": faiss.Index, "metadata": [{"doc_name": str, "chunk_idx": int, "text": str}]}}
faiss_stores: Dict[int, dict] = {}

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.35
EMBEDDING_DIM = 1536


def get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        client = OpenAI(api_key=api_key)
    return client


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    c = get_client()
    response = c.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype="float32")


def build_faiss_index(questionnaire_id: int, chunks: List[str], doc_name: str):
    """Add chunks to the FAISS index for a given questionnaire."""
    if not chunks:
        return

    embeddings = get_embeddings(chunks)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    if questionnaire_id not in faiss_stores:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        faiss_stores[questionnaire_id] = {"index": index, "metadata": []}

    store = faiss_stores[questionnaire_id]
    store["index"].add(embeddings)
    for i, chunk in enumerate(chunks):
        store["metadata"].append({
            "doc_name": doc_name,
            "chunk_idx": len(store["metadata"]),
            "text": chunk,
        })


def retrieve_chunks(questionnaire_id: int, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
    """Retrieve top-k most similar chunks for a query."""
    if questionnaire_id not in faiss_stores:
        return []

    store = faiss_stores[questionnaire_id]
    if store["index"].ntotal == 0:
        return []

    query_embedding = get_embeddings([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = store["index"].search(query_embedding, min(top_k, store["index"].ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(store["metadata"]):
            results.append((float(score), store["metadata"][idx]))
    return results


def generate_answer(question: str, questionnaire_id: int) -> Tuple[str, str]:
    """Generate an answer for a question using RAG.

    Returns: (answer, citation)
    """
    chunks = retrieve_chunks(questionnaire_id, question, top_k=3)

    # Check similarity threshold
    if not chunks or chunks[0][0] < SIMILARITY_THRESHOLD:
        return "Not found in references.", ""

    # Build context from retrieved chunks
    context_parts = []
    citations = []
    for score, meta in chunks:
        if score >= SIMILARITY_THRESHOLD:
            context_parts.append(meta["text"])
            citation = f"[{meta['doc_name']} - Chunk {meta['chunk_idx']}]"
            if citation not in citations:
                citations.append(citation)

    if not context_parts:
        return "Not found in references.", ""

    context = "\n\n---\n\n".join(context_parts)
    citation_str = ", ".join(citations)

    prompt = f"""You must answer ONLY using the provided context.
If the answer is not present in the context, respond exactly with:
'Not found in references.'

Context:
{context}

Question: {question}

Answer (include relevant details from the context):"""

    c = get_client()
    response = c.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise answering assistant. Answer ONLY from the given context. If the answer is not in the context, say exactly: 'Not found in references.'"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    answer = response.choices[0].message.content.strip()
    return answer, citation_str
