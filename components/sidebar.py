"""
components/sidebar.py — Collapsible sidebar with navigation.
"""

from __future__ import annotations
import streamlit as st


NAV_ITEMS = [
    ("💬", "Chat",           "chat"),
    ("📚", "Document Library","library"),
    ("🔬", "Chunk Explorer",  "explorer"),
    ("📊", "Analytics",       "analytics"),
    ("⚙️",  "Settings",        "settings"),
]


def render_sidebar() -> str:
    """Render the sidebar and return the active page key."""
    with st.sidebar:
        # Brand
        st.markdown(
            """
            <div class="brand-logo">
              <div class="brand-icon">🧠</div>
              <div>
                <div class="brand-name">DocIntel</div>
                <div class="brand-tagline">Semantic RAG Platform</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown("### Navigation")
        if "active_page" not in st.session_state:
            st.session_state.active_page = "chat"

        for icon, label, key in NAV_ITEMS:
            is_active = st.session_state.active_page == key
            css_class = "nav-item active" if is_active else "nav-item"
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_page = key
                st.rerun()

        st.divider()

        # Quick stats in sidebar
        if "analytics_cache" in st.session_state:
            snap = st.session_state.analytics_cache
            st.markdown("**Indexed**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Docs", snap.total_documents)
            with col2:
                st.metric("Chunks", snap.total_chunks)

        st.divider()
        st.caption("Powered by Gemini · Qdrant · Inngest")

    return st.session_state.active_page

