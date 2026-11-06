"""
streamlit_app.py — DocIntel: Semantic RAG Document Intelligence Platform.

Entry point for the Streamlit frontend.
Architecture: Streamlit → Inngest events → FastAPI/uvicorn worker.
"""

from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocIntel — Semantic RAG Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Imports (after set_page_config)
# ─────────────────────────────────────────────────────────────────────────────
from components.styles import GLOBAL_CSS
from components.sidebar import render_sidebar
from components.chat_view import render_chat
from components.document_library import render_library
from components.chunk_explorer import render_explorer
from components.analytics_view import render_analytics
from components.settings_view import render_settings


# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Analytics cache
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=60, show_spinner=False)
def _load_analytics():
    try:
        from vector_db import QdrantStorage
        store = QdrantStorage()
        return store.get_analytics()
    except Exception as exc:
        st.warning(f"Analytics unavailable: {exc}", icon="⚠️")
        return None


def _refresh_analytics():
    _load_analytics.clear()
    if "analytics_cache" in st.session_state:
        del st.session_state["analytics_cache"]


# ─────────────────────────────────────────────────────────────────────────────
# Pages loader for chunk comparison
# ─────────────────────────────────────────────────────────────────────────────


def _load_pages_for_source(source_id: str):
    """Load PDF pages for a given source_id from uploads, test doc, or workspace."""
    from data_loader import load_pdf_pages
    safe_name = Path(source_id).name
    candidates = [
        Path("uploads") / safe_name,
        Path("test doc") / safe_name,
        Path(source_id),
        Path(os.getcwd()) / safe_name,
    ]
    for c in candidates:
        if c.exists():
            return load_pdf_pages(str(c))
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant helpers for explorer
# ─────────────────────────────────────────────────────────────────────────────


def _get_chunks_for_source(source_id: str):
    try:
        from vector_db import QdrantStorage
        return QdrantStorage().get_chunks_for_source(source_id)
    except Exception as exc:
        st.error(f"Failed to load chunks: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Load analytics
    analytics = _load_analytics()
    if analytics is not None:
        st.session_state.analytics_cache = analytics

    # Available sources
    available_sources: list[str] = []
    if analytics and analytics.documents:
        available_sources = [d.source for d in analytics.documents]

    # Sidebar navigation
    active_page = render_sidebar()

    # ── Route to active page ────────────────────────────────────────
    if active_page == "chat":
        render_chat(available_sources)

    elif active_page == "library":
        render_library(analytics, _refresh_analytics)

    elif active_page == "explorer":
        from data_loader import embed_text
        render_explorer(
            analytics=analytics,
            get_chunks_fn=_get_chunks_for_source,
            available_sources=available_sources,
            pages_fn=_load_pages_for_source,
            embed_fn=embed_text,
        )

    elif active_page == "analytics":
        render_analytics(analytics, _refresh_analytics)

    elif active_page == "settings":
        render_settings()


if __name__ == "__main__":
    main()
else:
    main()
