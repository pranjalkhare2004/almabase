"""RAG Orchestrator — single clean pipeline for answering questions.

Flow:
  1. Retrieve + filter chunks (from pgvector)
  2. Log observability data
  3. Hard fallback gate (no LLM call if retrieval fails)
  4. Build context from selected chunks
  5. LLM generates answer text only
  6. Respect LLM fallback
  7. Attach citations programmatically
"""
import logging
from typing import Tuple, List

from sqlalchemy.orm import Session

from app.rag.retrieval import retrieve, ABSOLUTE_MIN_THRESHOLD, DYNAMIC_MARGIN
from app.rag.generation import generate
from app.rag.citation import attach_citations

logger = logging.getLogger("rag.orchestrator")

FALLBACK_ANSWER = "Not found in references."


def _log_retrieval(
    question: str,
    retrieved: List[Tuple[float, dict]],
    selected: List[Tuple[float, dict]],
    decision: str,
):
    """Structured observability logging for every question."""
    logger.info("=" * 72)
    logger.info("QUESTION: %s", question[:100])

    if not retrieved:
        logger.info("  ⚠ NO VECTORS in index")
        return

    best = retrieved[0][0]
    cutoff = best - DYNAMIC_MARGIN

    for rank, (score, meta) in enumerate(retrieved):
        passed = "✓" if score >= cutoff and best >= ABSOLUTE_MIN_THRESHOLD else "✗"
        logger.info(
            "  [%s] rank=%d  sim=%.4f  doc='%s'  chunk=%d  text='%s'",
            passed, rank, score, meta["doc_name"], meta["chunk_id"],
            meta["text"][:70].replace("\n", " "),
        )

    logger.info(
        "  SCORES: best=%.4f  cutoff=%.4f (best-%.2f)  floor=%.4f  decision=%s",
        best, cutoff, DYNAMIC_MARGIN, ABSOLUTE_MIN_THRESHOLD, decision,
    )

    if selected:
        logger.info(
            "  SELECTED %d chunks | scores=%s",
            len(selected), [f"{s:.4f}" for s, _ in selected],
        )


def answer_question(question: str, questionnaire_id: int, db: Session) -> Tuple[str, str]:
    """Full deterministic RAG pipeline for one question.

    Returns: (answer_text, citation_string)
    """
    # Step 1: Retrieve + filter
    retrieved, selected, decision = retrieve(questionnaire_id, question, db)

    # Step 2: Observability
    _log_retrieval(question, retrieved, selected, decision)

    # Step 3: Hard fallback gate
    if decision == "fallback":
        best = retrieved[0][0] if retrieved else 0.0
        logger.info(
            "  ✗ FALLBACK | best=%.4f | NO LLM call | NO citations",
            best,
        )
        return FALLBACK_ANSWER, ""

    # Step 4: Build context
    context = "\n\n---\n\n".join(meta["text"] for _, meta in selected)

    # Step 5: Generate answer (text only)
    answer, elapsed = generate(context, question)

    # Step 6: Respect LLM fallback
    if "not found in references" in answer.lower():
        logger.info("  LLM returned fallback → NO citations")
        return FALLBACK_ANSWER, ""

    # Step 7: Attach citations deterministically
    citations = attach_citations(selected)
    logger.info("  CITATIONS: %s", citations)
    logger.info("=" * 72)

    return answer, citations
