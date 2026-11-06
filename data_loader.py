"""
data_loader.py — Document extraction and embedding generation.

Responsibilities:
- Load PDFs via LlamaIndex PDFReader (preserved)
- Convert to PageContent objects with page metadata
- embed_text: batch Gemini embedding with task_type support, rate limiting, and exponential backoff
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted  # FIX: Gemini 429 Quota - import ResourceExhausted exception
import google.generativeai as genai
from llama_index.readers.file import PDFReader

from chunking_engine import PageContent

load_dotenv()

logger = logging.getLogger(__name__)

# FIX: API Key Fallback - Support both GEMINI_API_KEY and GOOGLE_API_KEY
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 3072


# ─────────────────────────────────────────────────────────────────────────────
# PDF loading
# ─────────────────────────────────────────────────────────────────────────────


def load_pdf_pages(path: str | Path) -> list[PageContent]:
    """
    Load a PDF and return one PageContent per page, preserving page numbers
    and file-name metadata as provided by LlamaIndex PDFReader.

    PDFReader returns one Document per page; metadata contains:
      - 'page_label' (str, e.g. '1')
      - 'file_name'  (str)
    """
    path = Path(path)
    docs = PDFReader().load_data(file=path)
    pages: list[PageContent] = []

    for doc in docs:
        text = getattr(doc, "text", None) or ""
        if not text.strip():
            continue
        metadata = doc.metadata or {}
        try:
            page_number = int(metadata.get("page_label", 1))
        except (ValueError, TypeError):
            page_number = len(pages) + 1

        pages.append(
            PageContent(
                page_number=page_number,
                text=text,
                file_name=metadata.get("file_name", path.name),
                metadata=metadata,
            )
        )

    logger.info("Loaded %d pages from '%s'", len(pages), path.name)
    return pages


def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Legacy compatibility shim — returns plain text chunks via SentenceSplitter.
    Used only when the new chunking pipeline is bypassed.
    """
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    pages = load_pdf_pages(path)
    chunks: list[str] = []
    for page in pages:
        chunks.extend(splitter.split_text(page.text))
    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding generation
# ─────────────────────────────────────────────────────────────────────────────


def embed_text(
    texts: list[str],
    task_type: str = "retrieval_document",
    batch_size: int = 50,
) -> list[list[float]]:
    """
    Batch-embed texts using Gemini embedding model with rate-limiting and retry logic.

    Parameters
    ----------
    texts     : list of strings to embed
    task_type : 'retrieval_document' for indexing, 'retrieval_query' for search
    batch_size: number of texts per API call

    Returns
    -------
    list of float vectors, one per input text
    """
    if not texts:
        return []

    embeddings: list[list[float]] = []
    total_texts = len(texts)

    for i in range(0, total_texts, batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = None

        # FIX: Gemini 429 Quota - Adaptive backoff retry loop respecting server cooldown
        max_retries = 6
        backoff_delay = 10.0

        for attempt in range(1, max_retries + 1):
            try:
                response = genai.embed_content(
                    model=EMBED_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                batch_embeddings = response["embedding"]
                break
            except ResourceExhausted as exc:
                if attempt == max_retries:
                    logger.error(
                        "Rate limit exceeded (429 ResourceExhausted) on batch %d-%d after %d retries. Failing: %s",
                        i, min(i + batch_size - 1, total_texts - 1), max_retries, exc
                    )
                    raise
                # Check for server-recommended retry delay in error message
                wait_sec = backoff_delay
                msg = str(exc)
                match = re.search(r"retry in\s+([0-9.]+)\s*s", msg, re.IGNORECASE) or re.search(r"seconds:\s*(\d+)", msg)
                if match:
                    try:
                        wait_sec = max(wait_sec, float(match.group(1)) + 2.0)
                    except (ValueError, TypeError):
                        pass
                logger.warning(
                    "ResourceExhausted (429) rate limit hit on batch %d-%d. Retrying attempt %d/%d after %.2fs backoff.",
                    i, min(i + batch_size - 1, total_texts - 1), attempt, max_retries, wait_sec
                )
                time.sleep(wait_sec)
                backoff_delay = max(wait_sec * 1.3, backoff_delay * 1.5)
            except Exception as exc:
                logger.error(
                    "Embedding batch %d-%d failed with unexpected error: %s",
                    i, min(i + batch_size - 1, total_texts - 1), exc
                )
                raise

        # FIX: Gemini 429 Quota - Safely handle both batch responses (list of lists) and single-element returns
        if batch_embeddings is not None:
            if isinstance(batch_embeddings, list) and len(batch_embeddings) > 0:
                if isinstance(batch_embeddings[0], list):
                    embeddings.extend(batch_embeddings)
                else:
                    embeddings.append(batch_embeddings)
            else:
                logger.warning("Unexpected embedding response format for batch starting at %d: %s", i, batch_embeddings)

        # FIX: Gemini 429 Quota - Explicit throttle delay between batch executions to stay below 100 RPM ceiling
        if i + batch_size < total_texts:
            time.sleep(1.2)

    return embeddings
