"""Utility functions for text extraction and processing."""
import re
from typing import List


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
    # Filter empty strings and strip whitespace
    questions = [q.strip() for q in parts if q.strip()]
    if len(questions) > 1:
        return questions
    # Fallback: split by blank lines or newlines
    questions = [q.strip() for q in text.split("\n") if q.strip()]
    return questions


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by approximate token count.

    Uses word-based splitting as a simple proxy for tokens.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
