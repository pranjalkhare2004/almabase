"""RAG package — modular retrieval-augmented generation pipeline."""
from app.rag.orchestrator import answer_question
from app.rag.retrieval import build_index

__all__ = ["answer_question", "build_index"]
