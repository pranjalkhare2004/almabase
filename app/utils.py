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
    # Try numbered pattern first: 1. or 1) or Q1. or Q1) etc.
    pattern = r'(?:^|\n)\s*(?:Q?\d+[\.\)]\s*)'
    parts = re.split(pattern, text)
    questions = [q.strip() for q in parts if q.strip()]
    if len(questions) > 1:
        return questions
    # Fallback: split by blank lines or newlines
    questions = [q.strip() for q in text.split("\n") if q.strip()]
    return questions


# --- Semantic-boundary chunking (Phase 1) ---

def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundaries."""
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 words per token (conservative)."""
    return max(1, int(len(text.split()) * 1.3))


def chunk_text(
    text: str,
    min_chunk_tokens: int = 400,
    max_chunk_tokens: int = 700,
    overlap_tokens: int = 75,
) -> List[str]:
    """Split text into chunks respecting sentence boundaries.

    Strategy:
      1. Split text into sentences
      2. Accumulate sentences into a chunk until max_chunk_tokens
      3. When a chunk reaches min_chunk_tokens and the next sentence
         would exceed max_chunk_tokens, finalize the chunk
      4. Start next chunk with overlap_tokens worth of trailing sentences

    This avoids splitting mid-sentence and keeps chunks 400-700 tokens.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)

        # If adding this sentence exceeds max and we already have enough
        if current_tokens + sent_tokens > max_chunk_tokens and current_tokens >= min_chunk_tokens:
            # Finalize current chunk
            chunk_text_str = " ".join(current_sentences)
            chunks.append(chunk_text_str)

            # Build overlap: take trailing sentences up to overlap_tokens
            overlap_sents: List[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tok = _estimate_tokens(s)
                if overlap_count + s_tok > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_tok

            current_sentences = overlap_sents
            current_tokens = overlap_count

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str)

    # Log chunk stats
    for i, chunk in enumerate(chunks):
        tok = _estimate_tokens(chunk)
        preview = chunk[:80].replace("\n", " ")
        logger.info("  Chunk %d: ~%d tokens, preview='%s...'", i, tok, preview)

    return chunks
