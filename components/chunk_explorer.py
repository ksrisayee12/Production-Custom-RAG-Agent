"""
components/chunk_explorer.py — Interactive Chunk Explorer with strategy comparison.
"""

from __future__ import annotations
from typing import Optional

import streamlit as st

from custom_types import ChunkMetadata, ChunkingStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _chunk_preview(text: str, max_chars: int = 140) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _render_chunk_card(chunk: ChunkMetadata, idx: int, selected: bool = False):
    """Render a single chunk card in the list."""
    css_class = "chunk-card selected" if selected else "chunk-card"
    strategy_badge = f'<span class="badge badge-info">{chunk.chunking_method}</span>'
    page_badge = (
        f'<span class="badge badge-muted">p.{chunk.page_number}</span>'
        if chunk.page_number else ""
    )
    token_badge = (
        f'<span class="badge badge-muted">{chunk.token_count} tokens</span>'
        if chunk.token_count else ""
    )
    section_badge = (
        f'<span class="badge badge-muted">§ {chunk.section[:30]}</span>'
        if chunk.section else ""
    )

    st.markdown(
        f"""
        <div class="{css_class}">
          <div class="chunk-index">Chunk #{chunk.chunk_index}</div>
          <div class="chunk-preview">{_chunk_preview(chunk.text)}</div>
          <div class="chunk-meta-row">
            {strategy_badge} {page_badge} {token_badge} {section_badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chunk_detail(chunk: ChunkMetadata, all_chunks: list[ChunkMetadata]):
    """Render the full detail panel for a selected chunk."""
    st.markdown("#### Chunk Detail")

    # Header metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunk Index", chunk.chunk_index)
    with col2:
        st.metric("Token Count", chunk.token_count or "—")
    with col3:
        pages = "—"
        if chunk.page_start and chunk.page_end and chunk.page_start != chunk.page_end:
            pages = f"pp.{chunk.page_start}–{chunk.page_end}"
        elif chunk.page_number:
            pages = f"p.{chunk.page_number}"
        st.metric("Pages", pages)

    # Section hierarchy
    if chunk.section or chunk.subsection:
        st.markdown("**Section Hierarchy**")
        if chunk.section:
            st.markdown(f"📂 **Section:** {chunk.section}")
        if chunk.subsection:
            st.markdown(f"  📄 **Subsection:** {chunk.subsection}")

    # Parent context
    if chunk.parent_context:
        st.markdown(f"**Context:** `{chunk.parent_context}`")

    # Semantic score
    if chunk.semantic_score is not None:
        st.markdown("**Semantic Boundary Score**")
        score = chunk.semantic_score
        bar_width = max(0, min(100, int(score * 100)))
        st.markdown(
            f"""
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:{bar_width}%"></div>
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:3px;">
              Avg internal similarity: {score:.3f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Strategy
    doc_id_preview = f"{chunk.document_id[:16]}…" if chunk.document_id else "—"
    st.markdown(
        f"**Chunking Method:** `{chunk.chunking_method}`  "
        f"**Document ID:** `{doc_id_preview}`"
    )

    st.divider()

    # Full text
    st.markdown("**Full Text**")
    st.markdown(
        f"""
        <div class="card" style="font-size:0.85rem;line-height:1.7;color:var(--text-primary);">
          {chunk.text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Neighboring chunks
    id_to_chunk = {c.chunk_id: c for c in all_chunks}

    prev_chunk: Optional[ChunkMetadata] = (
        id_to_chunk.get(chunk.previous_chunk_id) if chunk.previous_chunk_id else None
    )
    next_chunk: Optional[ChunkMetadata] = (
        id_to_chunk.get(chunk.next_chunk_id) if chunk.next_chunk_id else None
    )

    if prev_chunk or next_chunk:
        st.markdown("**Neighboring Chunks**")
        ncols = st.columns(2)
        with ncols[0]:
            if prev_chunk:
                st.markdown("← **Previous**")
                st.caption(f"#{prev_chunk.chunk_index} · {prev_chunk.token_count or '?'} tokens")
                st.markdown(
                    f"<div class='card-sm' style='font-size:0.78rem;color:var(--text-secondary);'>"
                    f"{_chunk_preview(prev_chunk.text, 200)}</div>",
                    unsafe_allow_html=True,
                )
        with ncols[1]:
            if next_chunk:
                st.markdown("→ **Next**")
                st.caption(f"#{next_chunk.chunk_index} · {next_chunk.token_count or '?'} tokens")
                st.markdown(
                    f"<div class='card-sm' style='font-size:0.78rem;color:var(--text-secondary);'>"
                    f"{_chunk_preview(next_chunk.text, 200)}</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Comparison view
# ─────────────────────────────────────────────────────────────────────────────


def _render_strategy_comparison(source_id: str, pages, embed_fn):
    """Run multiple chunking strategies and compare side-by-side."""
    st.markdown("### Strategy Comparison")
    st.info(
        "This runs **live chunking** on the selected document using different strategies. "
        "Semantic strategies require embedding API calls and may take a moment.",
        icon="⚡",
    )

    from chunking_engine import chunk_document, PageContent
    from custom_types import ChunkingConfig

    strategies_to_compare = [
        ChunkingStrategy.SENTENCE,
        ChunkingStrategy.STRUCTURE_AWARE,
        ChunkingStrategy.HYBRID_SEMANTIC,
    ]

    selected_strategies = st.multiselect(
        "Select strategies to compare",
        options=[s.value for s in strategies_to_compare],
        default=[ChunkingStrategy.SENTENCE.value, ChunkingStrategy.HYBRID_SEMANTIC.value],
        key="compare_strategies",
    )

    if not selected_strategies:
        return

    if st.button("▶ Run Comparison", type="primary", key="run_compare"):
        # Gemini Free-tier quota protection: sample pages if document is large
        comp_pages = pages
        if len(pages) > 3:
            st.info(
                f"💡 Running strategy comparison on a 3-page sample (pages 1–3 of {len(pages)}) "
                "to preserve Gemini API rate limits."
            )
            comp_pages = pages[:3]

        results = {}
        for strat_val in selected_strategies:
            strategy = ChunkingStrategy(strat_val)
            with st.spinner(f"Running `{strat_val}`…"):
                try:
                    needs_embed = strategy in (
                        ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID_SEMANTIC
                    )
                    result = chunk_document(
                        pages=comp_pages,
                        source_id=source_id + f"_compare_{strat_val}",
                        config=ChunkingConfig(strategy=strategy),
                        embed_fn=embed_fn if needs_embed else None,
                    )
                    results[strat_val] = result
                except Exception as exc:
                    st.error(f"`{strat_val}` failed: {exc}")

        if results:
            # Metrics comparison row
            st.markdown("#### Summary")
            metric_cols = st.columns(len(results))
            for col, (strat_val, res) in zip(metric_cols, results.items()):
                with col:
                    # FIX: NoneType String Formatting Crash - guard against NoneType for avg_chunk_tokens
                    avg_val = res.avg_chunk_tokens if res.avg_chunk_tokens is not None else 0.0
                    total_chunks_val = res.total_chunks if res.total_chunks is not None else 0
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-value'>{total_chunks_val}</div>"
                        f"<div class='metric-label'>{strat_val} chunks</div>"
                        f"<div style='font-size:0.72rem;color:var(--text-muted);margin-top:6px;'>"
                        f"avg {avg_val:.0f} tokens"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.divider()

            # Side-by-side first 5 chunks
            st.markdown("#### First 5 Chunks Per Strategy")
            tab_labels = list(results.keys())
            tabs = st.tabs(tab_labels)
            for tab, (strat_val, res) in zip(tabs, results.items()):
                with tab:
                    if not res.chunks:
                        st.info(f"No chunks generated for {strat_val}.")
                    else:
                        for chunk in res.chunks[:5]:
                            page_info = f"p.{chunk.page_number}" if chunk.page_number else ""
                            section_info = f"§ {chunk.section}" if chunk.section else ""
                            st.markdown(
                                f"<div class='chunk-card'>"
                                f"<div class='chunk-index'>Chunk #{chunk.chunk_index} · "
                                f"{chunk.token_count or '?'} tokens · {page_info} {section_info}</div>"
                                f"<div class='chunk-preview'>{chunk.text[:400]}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────


def render_explorer(
    analytics,
    get_chunks_fn,
    available_sources: list[str],
    pages_fn=None,
    embed_fn=None,
):
    """
    Render the full Chunk Explorer.

    Parameters
    ----------
    analytics         : AnalyticsSnapshot
    get_chunks_fn     : callable(source_id) -> list[ChunkMetadata]
    available_sources : list of indexed source IDs
    pages_fn          : callable(source_id) -> list[PageContent] (for comparison)
    embed_fn          : callable for embedding (for comparison)
    """
    st.markdown("## 🔬 Chunk Explorer")

    if not available_sources:
        st.info("No indexed documents found. Upload a PDF in Document Library.")
        return

    # Document selector
    preselect = st.session_state.pop("explorer_source", None)
    default_idx = 0
    if preselect and preselect in available_sources:
        default_idx = available_sources.index(preselect)

    selected_source = st.selectbox(
        "Select Document",
        options=available_sources,
        index=default_idx,
        key="explorer_source_select",
    )

    if not selected_source:
        return

    # Load chunks
    with st.spinner("Loading chunks…"):
        chunks = get_chunks_fn(selected_source)

    if not chunks:
        st.warning(f"No chunks found for `{selected_source}`.")
        return

    # ── Statistics ──────────────────────────────────────────────────
    st.markdown("### Statistics")
    total_chunks = len(chunks)
    token_counts = [c.token_count for c in chunks if c.token_count is not None]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    max_tokens = max(token_counts, default=0)
    min_tokens = min(token_counts, default=0)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{total_chunks}</div>"
            f"<div class='metric-label'>Total Chunks</div></div>",
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{avg_tokens:.0f}</div>"
            f"<div class='metric-label'>Avg Tokens</div></div>",
            unsafe_allow_html=True,
        )
    with sc3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{min_tokens}</div>"
            f"<div class='metric-label'>Min Tokens</div></div>",
            unsafe_allow_html=True,
        )
    with sc4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{max_tokens}</div>"
            f"<div class='metric-label'>Max Tokens</div></div>",
            unsafe_allow_html=True,
        )

    # Token distribution chart
    if token_counts:
        import pandas as pd
        df = pd.DataFrame({"tokens": token_counts, "chunk": list(range(len(token_counts)))})
        st.markdown("**Token Distribution per Chunk**")
        st.bar_chart(df.set_index("chunk")["tokens"], height=140, color="#63b3ed")

    st.divider()

    # ── Main explorer (list + detail) ───────────────────────────────
    main_tab, compare_tab = st.tabs(["📋 Browse Chunks", "⚖️ Compare Strategies"])

    with main_tab:
        if "explorer_selected_chunk_id" not in st.session_state:
            st.session_state.explorer_selected_chunk_id = chunks[0].chunk_id if chunks else None
        if "explorer_selected_idx" not in st.session_state:
            st.session_state.explorer_selected_idx = 0

        list_col, detail_col = st.columns([2, 3])

        with list_col:
            st.markdown("**Chunk List**")

            # Filter controls
            page_filter = st.number_input(
                "Filter by page (0 = all)", min_value=0, value=0, step=1, key="exp_page_filter"
            )
            search_term = st.text_input("Search text", placeholder="Filter by keyword…", key="exp_search")

            filtered = chunks
            if page_filter > 0:
                filtered = [c for c in chunks if c.page_number == page_filter]
            if search_term:
                filtered = [c for c in filtered if search_term.lower() in c.text.lower()]

            st.caption(f"Showing {len(filtered)} of {total_chunks} chunks")

            for i, chunk in enumerate(filtered[:100]):  # cap at 100 for perf
                is_sel = (
                    st.session_state.get("explorer_selected_chunk_id") == chunk.chunk_id
                    or (
                        st.session_state.get("explorer_selected_chunk_id") is None
                        and st.session_state.get("explorer_selected_idx") == chunk.chunk_index
                    )
                )
                _render_chunk_card(chunk, i, selected=is_sel)
                if st.button(
                    "Select",
                    key=f"sel_chunk_{i}_{chunk.chunk_id}",
                    use_container_width=True,
                ):
                    st.session_state.explorer_selected_chunk_id = chunk.chunk_id
                    st.session_state.explorer_selected_idx = chunk.chunk_index
                    st.rerun()

        with detail_col:
            sel_chunk_id = st.session_state.get("explorer_selected_chunk_id")
            sel_idx = st.session_state.get("explorer_selected_idx", 0)
            sel_chunk = None
            if sel_chunk_id:
                sel_chunk = next((c for c in chunks if c.chunk_id == sel_chunk_id), None)
            if sel_chunk is None:
                sel_chunk = next((c for c in chunks if c.chunk_index == sel_idx), None)
            if sel_chunk is None and chunks:
                sel_chunk = chunks[0]
            if sel_chunk:
                _render_chunk_detail(sel_chunk, chunks)

    with compare_tab:
        if pages_fn is not None and embed_fn is not None:
            pages = pages_fn(selected_source)
            _render_strategy_comparison(selected_source, pages, embed_fn)
        else:
            st.info("Comparison requires page loader and embedding function.")
