"""RAG module: embeddings, FAISS indexing, retrieval, answer generation via Groq."""
import os
from typing import List, Dict, Tuple, Optional
import numpy as np

import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

groq_client: Optional[Groq] = None
embed_model: Optional[SentenceTransformer] = None

# In-memory FAISS store per questionnaire
# Structure: {questionnaire_id: {"index": faiss.Index, "metadata": [...]}}
faiss_stores: Dict[int, dict] = {}

CHAT_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
SIMILARITY_THRESHOLD = 0.40


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
    scores, indices = store["index"].search(query_embedding, min(top_k, store["index"].ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(store["metadata"]):
            results.append((float(score), store["metadata"][idx]))
    return results


def generate_answer(question: str, questionnaire_id: int) -> Tuple[str, str]:
    """Generate an answer for a question using RAG with Groq.

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

    client = get_groq_client()
    response = client.chat.completions.create(
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
