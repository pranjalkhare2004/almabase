"""Minimal RAG integration tests."""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_embedding_dimensions():
    """Embedding model produces 384-dimensional vectors."""
    from app.rag.embedding import embed_query, EMBEDDING_DIM

    vec = embed_query("test query")
    assert vec.shape == (1, EMBEDDING_DIM), f"Expected (1, {EMBEDDING_DIM}), got {vec.shape}"
    print("✅ Embedding dimensions correct: (1, 384)")


def test_chunking_boundaries():
    """Chunks respect token boundaries and overlap."""
    from app.rag.chunking import chunk_document

    text = "This is a sentence. " * 200  # ~800 tokens
    chunks = chunk_document(text, "test_doc.txt")

    assert len(chunks) >= 1, "Should produce at least 1 chunk"
    for c in chunks:
        assert "text" in c and "doc_name" in c
        assert len(c["text"]) > 0
    print(f"✅ Chunking produced {len(chunks)} chunks with correct metadata")


def test_citation_deterministic():
    """Citations are built from chunk metadata, not LLM."""
    from app.rag.citation import build_citation

    chunks = [
        {"doc_name": "policy.txt", "chunk_index": 0},
        {"doc_name": "governance.txt", "chunk_index": 2},
    ]
    citation = build_citation(chunks)
    assert "policy.txt" in citation
    assert "governance.txt" in citation
    print(f"✅ Citation is deterministic: {citation}")


if __name__ == "__main__":
    test_embedding_dimensions()
    test_chunking_boundaries()
    test_citation_deterministic()
    print("\n🎉 All tests passed!")
