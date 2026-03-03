"""LLM answer generation via Groq.

Responsibility: Generate answer text ONLY.
The LLM must NOT produce citations, document names, or chunk IDs.
"""
import os
import time
import logging
from typing import Optional, Tuple

from groq import Groq

logger = logging.getLogger("rag.generation")

CHAT_MODEL = "llama-3.3-70b-versatile"

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = (
    "You are a precise answering assistant. "
    "Answer ONLY using the provided context. "
    "Provide concise but complete information. "
    "Do NOT invent information beyond what the context states. "
    "Do NOT include citations, references, document names, or source identifiers. "
    "If the context does not contain the answer, respond exactly: "
    "'Not found in references.'"
)


def generate(context: str, question: str) -> Tuple[str, float]:
    """Call LLM to produce answer text only.

    Args:
        context: concatenated text from selected chunks
        question: the user's question

    Returns:
        (answer_text, elapsed_seconds)
    """
    prompt = f"""Answer the following question using ONLY the provided context.
Provide a concise but complete answer.
Do not invent information beyond what is stated in the context.
If the answer is not in the context, respond exactly: 'Not found in references.'
Do NOT include citations or source references.

Context:
{context}

Question: {question}

Answer:"""

    client = _get_client()
    start = time.time()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=500,
    )
    elapsed = time.time() - start
    answer = resp.choices[0].message.content.strip()
    logger.info("  LLM: %.2fs | len=%d", elapsed, len(answer))
    return answer, elapsed
