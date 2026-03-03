"""Utility functions for text extraction and processing."""
import re
import logging
from typing import List

logger = logging.getLogger("utils")


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract plain text from uploaded file (PDF or TXT)."""
    if filename.lower().endswith(".txt"):
        return file_content.decode("utf-8", errors="ignore")
    elif filename.lower().endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def split_questions(text: str) -> List[str]:
    """Split text into individual questions by numbering or newlines."""
    pattern = r'(?:^|\n)\s*(?:Q?\d+[\.\)]\s*)'
    parts = re.split(pattern, text)
    questions = [q.strip() for q in parts if q.strip()]
    if len(questions) > 1:
        return questions
    questions = [q.strip() for q in text.split("\n") if q.strip()]
    return questions


# ---------------------------------------------------------------------------
# Paragraph-aware semantic chunking
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Conservative token estimate (~1.3 tokens per word)."""
    return max(1, int(len(text.split()) * 1.3))


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (double newline or indented blocks).
    Falls back to sentence splitting for long paragraphs.
    """
    # Split on double newlines first (standard paragraph separator)
    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # If no paragraph breaks found, try single newlines
    if len(paragraphs) <= 1 and text.strip():
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    return paragraphs


def _split_paragraph_into_sentences(paragraph: str) -> List[str]:
    """Split a paragraph into sentences."""
    raw = re.split(r'(?<=[.!?])\s+', paragraph)
    return [s.strip() for s in raw if s.strip()]


def chunk_text(
    text: str,
    min_chunk_tokens: int = 400,
    max_chunk_tokens: int = 700,
    overlap_tokens: int = 75,
) -> List[str]:
    """Chunk text respecting paragraph and sentence boundaries.

    Strategy:
      1. Split into paragraphs
      2. Accumulate paragraphs into chunks (400-700 tokens)
      3. If a single paragraph exceeds max, sub-split by sentences
      4. Add overlap from trailing content of previous chunk
      5. Each chunk is self-contained (no mid-sentence breaks)
    """
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    # Expand large paragraphs into sentence groups
    units: List[str] = []
    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if para_tokens <= max_chunk_tokens:
            units.append(para)
        else:
            # Break large paragraph into sentences
            sentences = _split_paragraph_into_sentences(para)
            for sent in sentences:
                units.append(sent)

    chunks: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _estimate_tokens(unit)

        # If adding this unit exceeds max and we already have enough content
        if current_tokens + unit_tokens > max_chunk_tokens and current_tokens >= min_chunk_tokens:
            chunk_str = "\n\n".join(current_units)
            chunks.append(chunk_str)

            # Build overlap from trailing units
            overlap_units: List[str] = []
            overlap_count = 0
            for u in reversed(current_units):
                u_tok = _estimate_tokens(u)
                if overlap_count + u_tok > overlap_tokens:
                    break
                overlap_units.insert(0, u)
                overlap_count += u_tok

            current_units = overlap_units
            current_tokens = overlap_count

        current_units.append(unit)
        current_tokens += unit_tokens

    # Final chunk
    if current_units:
        chunk_str = "\n\n".join(current_units)
        if chunk_str.strip():
            chunks.append(chunk_str)

    # Log chunk stats
    for i, chunk in enumerate(chunks):
        tok = _estimate_tokens(chunk)
        preview = chunk[:90].replace("\n", " ")
        logger.info("  chunk[%d]: ~%d tokens | '%s...'", i, tok, preview)
    logger.info("  Total chunks: %d", len(chunks))

    return chunks
