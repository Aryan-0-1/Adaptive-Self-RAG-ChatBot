"""
embeddings.py
-------------
Google Generative AI embeddings, with batched ingestion and
exponential backoff on transient rate-limit errors.

We switched away from local Ollama embeddings to Google's free-tier
embedding API so retrieval quality/setup doesn't depend on the user
having a capable local machine — but the free tier has a fairly low
per-minute quota, so large PDFs need to be embedded in batches with
backoff rather than in one giant call.

If `GOOGLE_API_KEY` isn't configured, `get_embeddings()` returns None
and callers (see vectorstore.py) treat that as "retrieval unavailable"
rather than crashing.
"""

from __future__ import annotations

import time
from typing import List, Optional

from langchain_core.documents import Document

from . import config

_embeddings = None
_embeddings_error: Optional[str] = None


def get_embeddings():
    """Lazily construct the embeddings client. Returns None if unconfigured."""
    global _embeddings, _embeddings_error

    if _embeddings is not None or _embeddings_error is not None:
        return _embeddings

    if not config.GOOGLE_API_KEY:
        _embeddings_error = "GOOGLE_API_KEY is not configured."
        return None

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _embeddings = GoogleGenerativeAIEmbeddings(
            model=config.GOOGLE_EMBED_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _embeddings_error = f"Failed to initialize Google embeddings: {exc}"
        _embeddings = None

    return _embeddings


def embeddings_error() -> Optional[str]:
    return _embeddings_error


def embed_documents_with_backoff(
    docs: List[Document],
    batch_size: int = None,
    max_retries: int = None,
) -> List[List[float]]:
    """
    Embed a list of Documents in batches, retrying each batch with
    exponential backoff on transient (quota/429-style) errors.

    Returns the flat list of embedding vectors, in the same order as
    `docs`. Raises the last exception if a batch never succeeds.
    """
    embeddings = get_embeddings()
    if embeddings is None:
        raise RuntimeError(embeddings_error() or "Embeddings not configured.")

    batch_size = batch_size or config.GOOGLE_EMBED_BATCH_SIZE
    max_retries = max_retries or config.GOOGLE_EMBED_MAX_RETRIES

    texts = [d.page_content for d in docs]
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        attempt = 0
        while True:
            try:
                vectors.extend(embeddings.embed_documents(batch))
                break
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    raise
                # Exponential backoff: 1s, 2s, 4s, ...
                time.sleep(2 ** (attempt - 1))

    return vectors
