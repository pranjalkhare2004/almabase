"""Local embedding using sentence-transformers.

Singleton model loader. No external API calls for embeddings.
"""
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Returns L2-normalized float32 array."""
    model = _get_model()
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns (1, EMBEDDING_DIM) array."""
    return embed_texts([query])
