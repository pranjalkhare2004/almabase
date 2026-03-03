"""Utility functions for text extraction and question parsing."""
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
    pattern = r'(?:^|\n)\s*(?:Q?\d+[\.\)]\s*)'
    parts = re.split(pattern, text)
    questions = [q.strip() for q in parts if q.strip()]
    if len(questions) > 1:
        return questions
    questions = [q.strip() for q in text.split("\n") if q.strip()]
    return questions
