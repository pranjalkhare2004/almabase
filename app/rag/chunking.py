"""Paragraph-aware document chunking.

Rules:
  - Chunk by paragraph boundaries first
  - Sub-split oversized paragraphs by sentence
  - Target: 400-700 tokens per chunk
  - Overlap: 75 tokens (trailing content from previous chunk)
  - Never split mid-sentence
  - Store metadata: doc_name, chunk_id, chunk_text
"""
import re
import logging
from typing import List

logger = logging.getLogger("rag.chunking")


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate (~1.3 tokens per word)."""
    return max(1, int(len(text.split()) * 1.3))


def _split_into_paragraphs(text: str) -> List[str]:
    """Split on double-newlines (paragraph boundaries).
    Falls back to single newlines if no paragraphs found.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(paragraphs) <= 1 and text.strip():
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences on punctuation boundaries."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]


def chunk_document(
    text: str,
    doc_name: str,
    min_tokens: int = 400,
    max_tokens: int = 700,
    overlap_tokens: int = 75,
) -> List[dict]:
    """Chunk a document into metadata-rich pieces.

    Returns list of dicts: {doc_name, chunk_id, text, token_count}
    """
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    # Break oversized paragraphs into sentences
    units: List[str] = []
    for para in paragraphs:
        if _estimate_tokens(para) <= max_tokens:
            units.append(para)
        else:
            for sent in _split_into_sentences(para):
                units.append(sent)

    # Accumulate units into chunks
    raw_chunks: List[str] = []
    current: List[str] = []
    current_tok = 0

    for unit in units:
        unit_tok = _estimate_tokens(unit)

        if current_tok + unit_tok > max_tokens and current_tok >= min_tokens:
            raw_chunks.append("\n\n".join(current))

            # Build overlap from trailing units
            overlap: List[str] = []
            ov_tok = 0
            for u in reversed(current):
                ut = _estimate_tokens(u)
                if ov_tok + ut > overlap_tokens:
                    break
                overlap.insert(0, u)
                ov_tok += ut

            current = overlap
            current_tok = ov_tok

        current.append(unit)
        current_tok += unit_tok

    if current:
        text_str = "\n\n".join(current)
        if text_str.strip():
            raw_chunks.append(text_str)

    # Build metadata-rich chunk list
    chunks: List[dict] = []
    for i, chunk_text in enumerate(raw_chunks):
        tok = _estimate_tokens(chunk_text)
        chunks.append({
            "doc_name": doc_name,
            "chunk_id": i,
            "text": chunk_text,
            "token_count": tok,
        })
        logger.info(
            "  chunk[%d] ~%d tokens | '%s...'",
            i, tok, chunk_text[:80].replace("\n", " "),
        )

    logger.info("Chunked '%s' → %d chunks", doc_name, len(chunks))
    return chunks
