"""
semantic_analyzer.py — Semantic similarity computation and boundary detection
for the hybrid chunking engine.

Responsibilities:
- Cosine similarity between sentence embedding vectors
- Adaptive boundary detection using local similarity distribution
- Structure-informed boundary boosting
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SentenceEmbedding:
    """A sentence paired with its embedding and structural context."""

    index: int
    text: str
    embedding: list[float]
    page_number: Optional[int] = None
    is_heading: bool = False
    structure_type: str = "paragraph"


@dataclass
class BoundaryAnalysis:
    """Semantic boundary information between two consecutive sentences."""

    after_index: int        # boundary exists after sentence[after_index]
    similarity: float       # cosine similarity with the next sentence
    is_boundary: bool = False
    boundary_strength: float = 0.0   # 0.0–1.0; higher = stronger boundary
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance (1 - cosine_similarity)."""
    return 1.0 - cosine_similarity(a, b)


def _moving_average(values: list[float], window: int = 3) -> list[float]:
    """Smoothed values using a centered moving average."""
    result = []
    half = window // 2
    n = len(values)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result.append(float(np.mean(values[start:end])))
    return result


def compute_pairwise_similarities(
    sentence_embeddings: list[SentenceEmbedding],
) -> list[float]:
    """
    Compute cosine similarity between each consecutive pair.
    Returns a list of length len(sentence_embeddings) - 1.
    """
    similarities: list[float] = []
    for i in range(len(sentence_embeddings) - 1):
        sim = cosine_similarity(
            sentence_embeddings[i].embedding,
            sentence_embeddings[i + 1].embedding,
        )
        similarities.append(sim)
    return similarities


def detect_boundaries(
    sentence_embeddings: list[SentenceEmbedding],
    similarities: list[float],
    base_threshold: float = 0.35,
    window_size: int = 5,
    heading_boost: float = 0.6,
) -> list[BoundaryAnalysis]:
    """
    Adaptive boundary detection using local similarity distribution.

    Strategy:
    1. Smooth similarities with a moving average.
    2. For each gap, compare smoothed similarity against local mean - std.
    3. Boost boundary confidence when a heading follows.
    4. Force boundaries at heading transitions and page transitions.

    Returns one BoundaryAnalysis per gap (len = len(sentences) - 1).
    """
    if len(similarities) == 0:
        return []

    smoothed = _moving_average(similarities, window=min(window_size, len(similarities)))
    n = len(smoothed)
    analyses: list[BoundaryAnalysis] = []

    # Pre-compute local statistics
    half_w = window_size // 2

    for i in range(n):
        sim = similarities[i]
        smoothed_sim = smoothed[i]

        # Local window for adaptive threshold
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        local_vals = smoothed[start:end]
        local_mean = float(np.mean(local_vals))
        local_std = float(np.std(local_vals)) if len(local_vals) > 1 else 0.0
        adaptive_threshold = local_mean - (0.5 * local_std)

        next_sentence = sentence_embeddings[i + 1]
        is_boundary = False
        strength = 0.0
        reason = ""

        # Rule 1: Heading following → force boundary
        if next_sentence.is_heading:
            is_boundary = True
            strength = heading_boost
            reason = "heading_transition"

        # Rule 2: Page change → strong boundary hint
        elif (
            sentence_embeddings[i].page_number is not None
            and next_sentence.page_number is not None
            and sentence_embeddings[i].page_number != next_sentence.page_number
        ):
            # Page change alone is not always a semantic break, but combines well
            if smoothed_sim < adaptive_threshold + 0.1:
                is_boundary = True
                strength = 0.4
                reason = "page_transition"

        # Rule 3: Low similarity vs. adaptive local threshold
        if not is_boundary and smoothed_sim < adaptive_threshold:
            is_boundary = True
            # Scale strength by how far below threshold we are
            gap = adaptive_threshold - smoothed_sim
            strength = min(1.0, gap / max(adaptive_threshold, 0.01))
            reason = f"similarity_drop(sim={smoothed_sim:.3f},thr={adaptive_threshold:.3f})"

        # Rule 4: Hard floor – always break at very low similarity
        if not is_boundary and sim < base_threshold:
            is_boundary = True
            strength = 0.3
            reason = f"hard_floor(sim={sim:.3f})"

        analyses.append(
            BoundaryAnalysis(
                after_index=i,
                similarity=sim,
                is_boundary=is_boundary,
                boundary_strength=strength,
                reason=reason,
            )
        )

    return analyses

