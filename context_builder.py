"""
context_builder.py — Context assembly and answer prompt construction.

Responsibilities:
- Receive retrieved chunks (RetrievedChunk list)
- Rank and deduplicate
- Group by document/section
- Expand neighboring chunks when budget allows
- Assemble a structured context string for the LLM
- Build the final prompt
"""

from __future__ import annotations

import logging
from typing import Optional

import tiktoken

from custom_types import (
    AssembledContext,
    RetrievedChunk,
    RetrievalConfig,
)

logger = logging.getLogger(__name__)

try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def _count_tokens(text: str) -> int:
    if _TOKENIZER is None:
        return len(text) // 4
    return len(_TOKENIZER.encode(text))


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────


def _deduplicate(
    chunks: list[RetrievedChunk], threshold: float = 0.92
) -> list[RetrievedChunk]:
    """
    Remove near-duplicate chunks by comparing text overlap.
    Two chunks are considered duplicates if token Jaccard similarity > threshold.
    """
    seen: list[RetrievedChunk] = []
    for candidate in chunks:
        cand_tokens = set(candidate.text.lower().split())
        is_dup = False
        for existing in seen:
            exist_tokens = set(existing.text.lower().split())
            if not cand_tokens or not exist_tokens:
                continue
            intersection = len(cand_tokens & exist_tokens)
            union = len(cand_tokens | exist_tokens)
            jaccard = intersection / union if union else 0.0
            if jaccard >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen.append(candidate)
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Context assembly
# ─────────────────────────────────────────────────────────────────────────────


def assemble_context(
    chunks: list[RetrievedChunk],
    config: Optional[RetrievalConfig] = None,
) -> AssembledContext:
    """
    Build a structured LLM context from retrieved chunks.

    Steps:
    1. Sort by score descending (primary retrieved chunks first)
    2. Deduplicate
    3. Group by source / section
    4. Assemble formatted context block within token budget
    5. Return AssembledContext

    Parameters
    ----------
    chunks : list of RetrievedChunk (may include neighbor chunks)
    config : RetrievalConfig for budget and dedup threshold
    """
    if config is None:
        config = RetrievalConfig()

    # Sort: primary chunks (non-neighbors) by score first, then neighbors
    primary = sorted(
        [c for c in chunks if not c.is_neighbor], key=lambda c: c.score, reverse=True
    )
    neighbors = [c for c in chunks if c.is_neighbor]

    ordered = primary + neighbors

    # Deduplicate
    ordered = _deduplicate(ordered, threshold=config.dedup_threshold)

    # Group by source
    source_groups: dict[str, list[RetrievedChunk]] = {}
    for chunk in ordered:
        source_groups.setdefault(chunk.source, []).append(chunk)

    # Sort within each source by chunk_index for reading order
    for source in source_groups:
        source_groups[source].sort(key=lambda c: c.chunk_index or 0)

    # Assemble text within token budget
    lines: list[str] = []
    used_chunks: list[RetrievedChunk] = []
    total_tokens = 0
    truncated = False

    for source, source_chunks in source_groups.items():
        # Section header
        source_header = f"\n[Source: {source}]"
        header_tokens = _count_tokens(source_header)

        if total_tokens + header_tokens > config.max_context_tokens:
            truncated = True
            break

        lines.append(source_header)
        total_tokens += header_tokens

        last_section: Optional[str] = None

        for chunk in source_chunks:
            # Sub-section header
            section_label = chunk.section or ""
            if section_label and section_label != last_section:
                sub_header = f"  [{section_label}]"
                sub_tokens = _count_tokens(sub_header)
                if total_tokens + sub_tokens <= config.max_context_tokens:
                    lines.append(sub_header)
                    total_tokens += sub_tokens
                last_section = section_label

            # Page reference
            page_ref = ""
            if chunk.page_number is not None:
                page_ref = f" (p.{chunk.page_number})"
            elif chunk.page_start is not None and chunk.page_end is not None:
                page_ref = f" (pp.{chunk.page_start}–{chunk.page_end})"

            chunk_text = chunk.text.strip()
            full_line = f"  {chunk_text}{page_ref}"
            chunk_tokens = _count_tokens(full_line)

            if total_tokens + chunk_tokens > config.max_context_tokens:
                truncated = True
                break

            lines.append(full_line)
            total_tokens += chunk_tokens
            used_chunks.append(chunk)

        if truncated:
            break

    context_text = "\n".join(lines).strip()
    sources = list(dict.fromkeys(c.source for c in used_chunks))

    return AssembledContext(
        context_text=context_text,
        chunks_used=used_chunks,
        sources=sources,
        token_count=total_tokens,
        truncated=truncated,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document analyst and question-answering assistant.

Your answers must be:
- Grounded exclusively in the retrieved document context provided below
- Precise, clear, and directly address the question
- Attributed to specific sources and page numbers when available
- Honest about uncertainty — if the context does not contain sufficient information to answer, say so explicitly

Do NOT:
- Invent facts not present in the context
- Present assumptions as document facts
- Answer from general knowledge when the context is insufficient

Format your response clearly. When referencing information, cite the source and page number."""


def build_prompt(question: str, context: AssembledContext) -> str:
    """
    Build the full prompt string for Gemini.
    """
    truncation_notice = ""
    if context.truncated:
        truncation_notice = (
            "\n[Note: The retrieved context was truncated to fit within the context limit. "
            "Some relevant information may not be included.]\n"
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Retrieved Context ---\n"
        f"{context.context_text}\n"
        f"{truncation_notice}"
        f"--- End of Context ---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

