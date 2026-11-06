"""
components/settings_view.py — Application settings and configuration panel.
"""

from __future__ import annotations
import streamlit as st


def render_settings():
    st.markdown("## ⚙️ Settings")

    st.markdown("### Chunking Defaults")

    from custom_types import ChunkingConfig, ChunkingStrategy

    if "settings_strategy" not in st.session_state:
        st.session_state.settings_strategy = ChunkingStrategy.HYBRID_SEMANTIC.value
    if "settings_chunk_size" not in st.session_state:
        st.session_state.settings_chunk_size = 1000
    if "settings_overlap" not in st.session_state:
        st.session_state.settings_overlap = 150
    if "settings_min_size" not in st.session_state:
        st.session_state.settings_min_size = 100
    if "settings_max_size" not in st.session_state:
        st.session_state.settings_max_size = 2000
    if "settings_threshold" not in st.session_state:
        st.session_state.settings_threshold = 0.35

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.settings_strategy = st.selectbox(
            "Default Chunking Strategy",
            options=[s.value for s in ChunkingStrategy],
            index=[s.value for s in ChunkingStrategy].index(st.session_state.settings_strategy),
            key="set_strategy",
        )
        st.session_state.settings_chunk_size = st.number_input(
            "Chunk Size (chars)", min_value=100, max_value=5000,
            value=st.session_state.settings_chunk_size, key="set_chunk_size"
        )
        st.session_state.settings_overlap = st.number_input(
            "Chunk Overlap (chars)", min_value=0, max_value=500,
            value=st.session_state.settings_overlap, key="set_overlap"
        )
    with col2:
        st.session_state.settings_min_size = st.number_input(
            "Min Chunk Size (chars)", min_value=10, max_value=500,
            value=st.session_state.settings_min_size, key="set_min_size"
        )
        st.session_state.settings_max_size = st.number_input(
            "Max Chunk Size (chars)", min_value=200, max_value=10000,
            value=st.session_state.settings_max_size, key="set_max_size"
        )
        st.session_state.settings_threshold = st.slider(
            "Semantic Boundary Threshold", min_value=0.1, max_value=0.8,
            value=st.session_state.settings_threshold, step=0.05, key="set_threshold"
        )

    st.divider()
    st.markdown("### Retrieval Defaults")

    if "settings_top_k" not in st.session_state:
        st.session_state.settings_top_k = 5
    if "settings_enable_neighbors" not in st.session_state:
        st.session_state.settings_enable_neighbors = True
    if "settings_max_context_tokens" not in st.session_state:
        st.session_state.settings_max_context_tokens = 8000

    rc1, rc2 = st.columns(2)
    with rc1:
        st.session_state.settings_top_k = st.number_input(
            "Default Top-K", min_value=1, max_value=30,
            value=st.session_state.settings_top_k, key="set_top_k"
        )
        st.session_state.settings_enable_neighbors = st.toggle(
            "Enable Neighbor Expansion",
            value=st.session_state.settings_enable_neighbors, key="set_neighbors"
        )
    with rc2:
        st.session_state.settings_max_context_tokens = st.number_input(
            "Max Context Tokens", min_value=1000, max_value=32000,
            value=st.session_state.settings_max_context_tokens, key="set_ctx_tokens"
        )

    st.divider()
    st.markdown("### Environment")
    st.info("API credentials are loaded from `.env`. Never expose them in the UI.", icon="🔒")

    import os
    env_vars = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "INNGEST_API_BASE": os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1"),
    }
    rows = []
    for key, val in env_vars.items():
        rows.append({"Variable": key, "Status": "✅ Set" if val else "❌ Missing"})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

