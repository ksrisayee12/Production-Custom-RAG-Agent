"""
chunking_engine.py — Modular chunking engine supporting multiple strategies.

Strategies available:
  - fixed          : fixed character-count windows
  - sentence       : sentence-boundary aware (LlamaIndex SentenceSplitter)
  - recursive      : recursive character splitting
  - structure_aware: respects heading/paragraph structure
  - semantic       : pure embedding-based semantic grouping
  - hybrid_semantic: combines structure detection + semantic similarity (DEFAULT)

Public API:
  chunk_document(pages, source_id, config) -> ChunkingResult
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nltk
import tiktoken

from custom_types import (
    ChunkMetadata,
    ChunkingConfig,
    ChunkingResult,
    ChunkingStrategy,
    StructureType,
)
from semantic_analyzer import (
    BoundaryAnalysis,
    SentenceEmbedding,
    compute_pairwise_similarities,
    cosine_similarity,
    detect_boundaries,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser — cl100k_base is a close proxy for Gemini token counts
# ─────────────────────────────────────────────────────────────────────────────
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    if _TOKENIZER is None:
        return len(text) // 4
    return len(_TOKENIZER.encode(text))


# ─────────────────────────────────────────────────────────────────────────────
# Page-level data class
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PageContent:
    """Represents a single page from the PDF."""

    page_number: int         # 1-based
    text: str
    file_name: str = ""
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Structure detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+"),                         # Markdown headings
    re.compile(r"^[A-Z][A-Z0-9 ,:\-]{2,60}$"),           # ALL-CAPS short lines
    re.compile(r"^\d+\.\s+[A-Z].{3,80}$"),               # Numbered section headers
    re.compile(r"^[A-Z].{3,60}:$"),                      # Title ending with colon
    re.compile(r"^(?:Chapter|Section|Part|Appendix)\s+", re.IGNORECASE),
]

_CODE_BLOCK_PATTERN = re.compile(r"^```|^~~~", re.MULTILINE)

_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-•*·]\s+|\d+[.)]\s+)")

_DEFINITION_PATTERN = re.compile(
    r"(?:is defined as|refers to|means|is the|can be defined)", re.IGNORECASE
)


def detect_structure_type(line: str) -> StructureType:
    stripped = line.strip()
    if not stripped:
        return StructureType.UNKNOWN

    # Check list items BEFORE heading patterns to avoid numbered list items
    # being classified as headings
    if _LIST_ITEM_PATTERN.match(stripped):
        if re.match(r"^\s*\d+[.)]\s+", stripped):
            return StructureType.NUMBERED_LIST
        return StructureType.LIST_ITEM

    for pat in _HEADING_PATTERNS:
        if pat.match(stripped):
            return StructureType.HEADING

    if _CODE_BLOCK_PATTERN.match(stripped):
        return StructureType.CODE_BLOCK

    if _DEFINITION_PATTERN.search(stripped):
        return StructureType.DEFINITION

    return StructureType.PARAGRAPH


def is_heading(line: str) -> bool:
    return detect_structure_type(line) == StructureType.HEADING


# ─────────────────────────────────────────────────────────────────────────────
# Sentence segmentation
# ─────────────────────────────────────────────────────────────────────────────


def segment_sentences(text: str) -> list[str]:
    """Split text into meaningful sentences using NLTK punkt."""
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except Exception:
        # Fallback: split on '. ', '! ', '? '
        sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter out empty / whitespace-only
    return [s.strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Internal paragraph/section extraction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TextBlock:
    text: str
    structure_type: StructureType
    page_number: int
    section: Optional[str] = None
    subsection: Optional[str] = None


def extract_blocks(pages: list[PageContent]) -> list[TextBlock]:
    """
    Convert page texts into structured blocks preserving heading hierarchy.
    """
    blocks: list[TextBlock] = []
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    for page in pages:
        lines = page.text.split("\n")
        paragraph_lines: list[str] = []

        def flush_paragraph():
            nonlocal paragraph_lines
            text = " ".join(paragraph_lines).strip()
            if text:
                blocks.append(
                    TextBlock(
                        text=text,
                        structure_type=StructureType.PARAGRAPH,
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                    )
                )
            paragraph_lines = []

        in_code_block = False
        code_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Code block toggle
            if _CODE_BLOCK_PATTERN.match(stripped):
                if in_code_block:
                    code_lines.append(line)
                    blocks.append(
                        TextBlock(
                            text="\n".join(code_lines),
                            structure_type=StructureType.CODE_BLOCK,
                            page_number=page.page_number,
                            section=current_section,
                            subsection=current_subsection,
                        )
                    )
                    code_lines = []
                    in_code_block = False
                else:
                    flush_paragraph()
                    in_code_block = True
                    code_lines = [line]
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            if not stripped:
                flush_paragraph()
                continue

            stype = detect_structure_type(stripped)

            if stype == StructureType.HEADING:
                flush_paragraph()
                # Determine heading level
                heading_text = stripped.lstrip("#").strip().rstrip(":")
                # Heuristic: short headings at top become sections; otherwise subsections
                if len(heading_text) < 50 and current_section is None:
                    current_section = heading_text
                    current_subsection = None
                elif current_section and heading_text != current_section:
                    current_subsection = heading_text
                else:
                    current_section = heading_text
                    current_subsection = None

                blocks.append(
                    TextBlock(
                        text=stripped,
                        structure_type=StructureType.HEADING,
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                    )
                )

            elif stype in (StructureType.LIST_ITEM, StructureType.NUMBERED_LIST):
                flush_paragraph()
                blocks.append(
                    TextBlock(
                        text=stripped,
                        structure_type=stype,
                        page_number=page.page_number,
                        section=current_section,
                        subsection=current_subsection,
                    )
                )

            else:
                paragraph_lines.append(stripped)

        flush_paragraph()

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-size chunking
# ─────────────────────────────────────────────────────────────────────────────


def _fixed_chunks(
    pages: list[PageContent], config: ChunkingConfig, doc_id: str, source: str
) -> list[ChunkMetadata]:
    results = []
    idx = 0
    for page in pages:
        text = page.text
        step = config.chunk_size - config.chunk_overlap
        for start in range(0, len(text), max(step, 1)):
            chunk_text = text[start : start + config.chunk_size].strip()
            if len(chunk_text) < config.min_chunk_size:
                continue
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
            results.append(
                ChunkMetadata(
                    document_id=doc_id,
                    source=source,
                    chunk_id=chunk_id,
                    chunk_index=idx,
                    text=chunk_text,
                    page_number=page.page_number,
                    chunking_method=ChunkingStrategy.FIXED.value,
                    token_count=count_tokens(chunk_text),
                )
            )
            idx += 1
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-based chunking
# ─────────────────────────────────────────────────────────────────────────────


def _sentence_chunks(
    pages: list[PageContent], config: ChunkingConfig, doc_id: str, source: str
) -> list[ChunkMetadata]:
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    results = []
    idx = 0
    for page in pages:
        for chunk_text in splitter.split_text(page.text):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < config.min_chunk_size:
                continue
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
            results.append(
                ChunkMetadata(
                    document_id=doc_id,
                    source=source,
                    chunk_id=chunk_id,
                    chunk_index=idx,
                    text=chunk_text,
                    page_number=page.page_number,
                    chunking_method=ChunkingStrategy.SENTENCE.value,
                    token_count=count_tokens(chunk_text),
                )
            )
            idx += 1
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Recursive chunking
# ─────────────────────────────────────────────────────────────────────────────


def _recursive_split(
    text: str, separators: list[str], max_size: int, overlap: int
) -> list[str]:
    if len(text) <= max_size:
        return [text] if text.strip() else []

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If part itself is too long, recurse with next separator
                    sub = _recursive_split(
                        part, separators[separators.index(sep) + 1 :], max_size, overlap
                    )
                    chunks.extend(sub)
                    current = ""
            if current:
                chunks.append(current)
            return chunks

    # No separator found → hard split
    result = []
    step = max(max_size - overlap, 1)
    for i in range(0, len(text), step):
        result.append(text[i : i + max_size])
    return result


def _recursive_chunks(
    pages: list[PageContent], config: ChunkingConfig, doc_id: str, source: str
) -> list[ChunkMetadata]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    results = []
    idx = 0
    for page in pages:
        raw_chunks = _recursive_split(
            page.text, separators, config.chunk_size, config.chunk_overlap
        )
        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < config.min_chunk_size:
                continue
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
            results.append(
                ChunkMetadata(
                    document_id=doc_id,
                    source=source,
                    chunk_id=chunk_id,
                    chunk_index=idx,
                    text=chunk_text,
                    page_number=page.page_number,
                    chunking_method=ChunkingStrategy.RECURSIVE.value,
                    token_count=count_tokens(chunk_text),
                )
            )
            idx += 1
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Structure-aware chunking
# ─────────────────────────────────────────────────────────────────────────────


def _structure_chunks(
    pages: list[PageContent], config: ChunkingConfig, doc_id: str, source: str
) -> list[ChunkMetadata]:
    blocks = extract_blocks(pages)
    groups: list[list[TextBlock]] = []
    current_group: list[TextBlock] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block.text)

        # Headings always start a new group
        if block.structure_type == StructureType.HEADING:
            if current_group:
                groups.append(current_group)
            current_group = [block]
            current_tokens = block_tokens
            continue

        if current_tokens + block_tokens > config.max_chunk_size and current_group:
            groups.append(current_group)
            current_group = [block]
            current_tokens = block_tokens
        else:
            current_group.append(block)
            current_tokens += block_tokens

    if current_group:
        groups.append(current_group)

    results = []
    idx = 0
    for group in groups:
        text = " ".join(b.text for b in group).strip()
        if len(text) < config.min_chunk_size:
            continue
        page_numbers = list({b.page_number for b in group})
        page_numbers.sort()
        section = next((b.section for b in group if b.section), None)
        subsection = next((b.subsection for b in group if b.subsection), None)
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
        results.append(
            ChunkMetadata(
                document_id=doc_id,
                source=source,
                chunk_id=chunk_id,
                chunk_index=idx,
                text=text,
                page_number=page_numbers[0] if page_numbers else None,
                page_start=page_numbers[0] if page_numbers else None,
                page_end=page_numbers[-1] if page_numbers else None,
                section=section,
                subsection=subsection,
                chunking_method=ChunkingStrategy.STRUCTURE_AWARE.value,
                token_count=count_tokens(text),
            )
        )
        idx += 1
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Semantic & Hybrid semantic chunking (core)
# ─────────────────────────────────────────────────────────────────────────────


def _build_sentence_embeddings(
    blocks: list[TextBlock],
    embed_fn,
    batch_size: int = 50,
) -> list[SentenceEmbedding]:
    """
    Expand blocks into individual sentences, then batch-embed them.
    Returns SentenceEmbedding list in order.
    """
    items: list[tuple[str, TextBlock]] = []
    for block in blocks:
        sentences = segment_sentences(block.text)
        for s in sentences:
            if s.strip():
                items.append((s, block))

    if not items:
        return []

    texts = [t for t, _ in items]

    # FIX: Batch Size Forwarding - explicitly forward batch_size to embed_fn for dynamic batch control
    try:
        all_embeddings: list[list[float]] = embed_fn(
            texts, task_type="retrieval_document", batch_size=batch_size
        )
    except TypeError:
        all_embeddings = embed_fn(texts, task_type="retrieval_document")

    result: list[SentenceEmbedding] = []
    for idx, ((text, block), emb) in enumerate(zip(items, all_embeddings)):
        result.append(
            SentenceEmbedding(
                index=idx,
                text=text,
                embedding=emb,
                page_number=block.page_number,
                is_heading=(block.structure_type == StructureType.HEADING),
                structure_type=block.structure_type.value,
            )
        )
    return result


def _group_sentences_into_chunks(
    sentence_embeddings: list[SentenceEmbedding],
    boundaries: list[BoundaryAnalysis],
    blocks: list[TextBlock],
    config: ChunkingConfig,
) -> list[list[SentenceEmbedding]]:
    """
    Group sentences into coherent semantic groups based on detected boundaries.
    Respects min/max chunk sizes.
    """
    groups: list[list[SentenceEmbedding]] = []
    current: list[SentenceEmbedding] = []
    current_tokens = 0

    for i, se in enumerate(sentence_embeddings):
        token_count = count_tokens(se.text)

        # Check if there's a boundary after the previous sentence
        is_split_point = False
        boundary_strength = 0.0
        if i > 0 and i - 1 < len(boundaries):
            ba = boundaries[i - 1]
            is_split_point = ba.is_boundary
            boundary_strength = ba.boundary_strength

        # Force split if adding this sentence would exceed max size
        if current_tokens + token_count > config.max_chunk_size and current:
            groups.append(current)
            # Apply overlap: carry last sentence(s) into next group
            overlap_sentences = _get_overlap_sentences(current, config.chunk_overlap)
            current = list(overlap_sentences)
            current_tokens = sum(count_tokens(s.text) for s in current)

        # Semantic boundary split (only if current group is large enough)
        elif is_split_point and current_tokens >= config.min_chunk_size:
            groups.append(current)
            # Apply overlap
            overlap_sentences = _get_overlap_sentences(current, config.chunk_overlap)
            current = list(overlap_sentences)
            current_tokens = sum(count_tokens(s.text) for s in current)

        current.append(se)
        current_tokens += token_count

    if current:
        groups.append(current)

    # Merge tiny groups with neighbors
    return _merge_tiny_groups(groups, config)


def _get_overlap_sentences(
    group: list[SentenceEmbedding], overlap_chars: int
) -> list[SentenceEmbedding]:
    """Return the last N sentences that fit within overlap_chars."""
    result: list[SentenceEmbedding] = []
    total = 0
    for se in reversed(group):
        total += len(se.text)
        if total > overlap_chars:
            break
        result.insert(0, se)
    return result


def _merge_tiny_groups(
    groups: list[list[SentenceEmbedding]],
    config: ChunkingConfig,
) -> list[list[SentenceEmbedding]]:
    """Merge consecutive groups that are individually below min_chunk_size."""
    if not groups:
        return groups

    merged: list[list[SentenceEmbedding]] = []
    buffer: list[SentenceEmbedding] = []
    buffer_tokens = 0

    for group in groups:
        group_tokens = sum(count_tokens(s.text) for s in group)

        # If merging buffer + group would exceed max, flush buffer first
        if buffer and buffer_tokens + group_tokens > config.max_chunk_size:
            if buffer_tokens >= config.min_chunk_size:
                merged.append(buffer)
            buffer = list(group)
            buffer_tokens = group_tokens
        else:
            buffer.extend(group)
            buffer_tokens += group_tokens

        # Flush buffer if it has reached min size and the group itself wasn't tiny
        if buffer_tokens >= config.min_chunk_size and group_tokens >= config.min_chunk_size:
            merged.append(buffer)
            buffer = []
            buffer_tokens = 0

    if buffer:
        if merged:
            # Append remainder to last group if possible
            last_tokens = sum(count_tokens(s.text) for s in merged[-1])
            if last_tokens + buffer_tokens <= config.max_chunk_size:
                merged[-1].extend(buffer)
            else:
                merged.append(buffer)
        else:
            merged.append(buffer)

    return merged


def _semantic_chunks_from_embeddings(
    pages: list[PageContent],
    embed_fn,
    config: ChunkingConfig,
    doc_id: str,
    source: str,
    strategy: ChunkingStrategy,
    use_structure: bool = False,
) -> list[ChunkMetadata]:
    """Core semantic/hybrid chunking implementation."""
    blocks = extract_blocks(pages)

    if not blocks:
        return []

    logger.info("Building sentence embeddings for %d blocks...", len(blocks))
    sentence_embeddings = _build_sentence_embeddings(
        blocks, embed_fn, batch_size=config.embedding_batch_size
    )

    if len(sentence_embeddings) < 2:
        # Fallback for very short documents
        text = " ".join(se.text for se in sentence_embeddings)
        if not text.strip():
            return []
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:0"))
        return [
            ChunkMetadata(
                document_id=doc_id,
                source=source,
                chunk_id=chunk_id,
                chunk_index=0,
                text=text.strip(),
                page_number=pages[0].page_number if pages else None,
                chunking_method=strategy.value,
                token_count=count_tokens(text),
            )
        ]

    logger.info("Computing pairwise similarities for %d sentences...", len(sentence_embeddings))
    similarities = compute_pairwise_similarities(sentence_embeddings)

    logger.info("Detecting boundaries...")
    boundaries = detect_boundaries(
        sentence_embeddings,
        similarities,
        base_threshold=config.semantic_threshold,
    )

    logger.info("Grouping sentences into chunks...")
    groups = _group_sentences_into_chunks(sentence_embeddings, boundaries, blocks, config)

    # Build ChunkMetadata objects
    results: list[ChunkMetadata] = []
    for idx, group in enumerate(groups):
        text = " ".join(se.text for se in group).strip()
        if len(text) < config.min_chunk_size:
            continue

        page_numbers = sorted({se.page_number for se in group if se.page_number is not None})
        # Find section/subsection from corresponding blocks
        block_section = None
        block_subsection = None
        for b in blocks:
            if any(se.text in b.text for se in group[:3]):
                block_section = b.section
                block_subsection = b.subsection
                break

        # Semantic score at this chunk boundary (average of boundary similarities in group)
        group_boundary_sims = [
            boundaries[se.index - 1].similarity
            for se in group
            if se.index > 0 and se.index - 1 < len(boundaries)
        ]
        avg_boundary_sim = (
            float(sum(group_boundary_sims) / len(group_boundary_sims))
            if group_boundary_sims
            else None
        )

        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
        results.append(
            ChunkMetadata(
                document_id=doc_id,
                source=source,
                chunk_id=chunk_id,
                chunk_index=idx,
                text=text,
                page_number=page_numbers[0] if page_numbers else None,
                page_start=page_numbers[0] if page_numbers else None,
                page_end=page_numbers[-1] if page_numbers else None,
                section=block_section,
                subsection=block_subsection,
                chunking_method=strategy.value,
                token_count=count_tokens(text),
                semantic_score=avg_boundary_sim,
            )
        )

    # Link consecutive chunks (prev/next IDs)
    for i, chunk in enumerate(results):
        if i > 0:
            chunk.previous_chunk_id = results[i - 1].chunk_id
        if i < len(results) - 1:
            chunk.next_chunk_id = results[i + 1].chunk_id

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Context enrichment
# ─────────────────────────────────────────────────────────────────────────────


def enrich_chunk_context(chunk: ChunkMetadata) -> ChunkMetadata:
    """
    Prepend a minimal context label to give retrieval more signal.
    The original text is preserved in chunk.text; parent_context holds the label.
    """
    parts = []
    if chunk.section:
        parts.append(f"Section: {chunk.section}")
    if chunk.subsection:
        parts.append(f"Subsection: {chunk.subsection}")
    if chunk.page_number:
        parts.append(f"Page {chunk.page_number}")

    if parts:
        chunk.parent_context = " | ".join(parts)

    return chunk


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def chunk_document(
    pages: list[PageContent],
    source_id: str,
    config: Optional[ChunkingConfig] = None,
    embed_fn=None,
) -> ChunkingResult:
    """
    Chunk a document (list of PageContent) according to config.strategy.

    Parameters
    ----------
    pages      : list of PageContent objects (1 per PDF page)
    source_id  : user-visible source identifier
    config     : ChunkingConfig; defaults to hybrid_semantic
    embed_fn   : callable(texts, task_type) -> list[list[float]]
                 required for semantic and hybrid_semantic strategies

    Returns
    -------
    ChunkingResult with full ChunkMetadata list
    """
    if config is None:
        config = ChunkingConfig()

    # Deterministic document ID based on source
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))

    strategy = config.strategy
    chunks: list[ChunkMetadata] = []

    logger.info("Chunking document '%s' with strategy '%s'", source_id, strategy.value)

    if strategy == ChunkingStrategy.FIXED:
        chunks = _fixed_chunks(pages, config, doc_id, source_id)

    elif strategy == ChunkingStrategy.SENTENCE:
        chunks = _sentence_chunks(pages, config, doc_id, source_id)

    elif strategy == ChunkingStrategy.RECURSIVE:
        chunks = _recursive_chunks(pages, config, doc_id, source_id)

    elif strategy == ChunkingStrategy.STRUCTURE_AWARE:
        chunks = _structure_chunks(pages, config, doc_id, source_id)

    elif strategy == ChunkingStrategy.SEMANTIC:
        if embed_fn is None:
            raise ValueError("embed_fn required for semantic chunking")
        chunks = _semantic_chunks_from_embeddings(
            pages, embed_fn, config, doc_id, source_id,
            strategy=ChunkingStrategy.SEMANTIC, use_structure=False
        )

    elif strategy == ChunkingStrategy.HYBRID_SEMANTIC:
        if embed_fn is None:
            raise ValueError("embed_fn required for hybrid_semantic chunking")
        chunks = _semantic_chunks_from_embeddings(
            pages, embed_fn, config, doc_id, source_id,
            strategy=ChunkingStrategy.HYBRID_SEMANTIC, use_structure=True
        )

    # Context enrichment pass
    if config.enable_context_enrichment:
        chunks = [enrich_chunk_context(c) for c in chunks]

    if not chunks:
        logger.warning("No chunks produced for document '%s'", source_id)

    avg_tokens: Optional[float] = None
    if chunks:
        token_counts = [c.token_count for c in chunks if c.token_count is not None]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else None

    return ChunkingResult(
        document_id=doc_id,
        source=source_id,
        strategy=strategy,
        chunks=chunks,
        total_chunks=len(chunks),
        avg_chunk_tokens=avg_tokens,
    )
