"""
components/analytics_view.py — Real analytics dashboard using Qdrant data.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_analytics(analytics, refresh_fn):
    st.markdown("## 📊 Analytics")

    if st.button("🔄 Refresh Analytics", key="analytics_refresh"):
        refresh_fn()
        st.rerun()

    if analytics is None:
        st.info("Loading analytics…")
        return

    # ── Top-level metrics ───────────────────────────────────────────
    st.markdown("### Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{analytics.total_documents}</div>"
            f"<div class='metric-label'>Documents</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{analytics.total_chunks:,}</div>"
            f"<div class='metric-label'>Total Chunks</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        avg_tokens = analytics.avg_chunk_tokens if analytics.avg_chunk_tokens is not None else 0.0
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{avg_tokens:.0f}</div>"
            f"<div class='metric-label'>Avg Tokens / Chunk</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Strategy distribution ───────────────────────────────────────
    if analytics.strategy_distribution:
        st.markdown("### Chunking Strategy Distribution")
        strat_df = pd.DataFrame(
            [
                {"Strategy": k, "Chunks": v}
                for k, v in analytics.strategy_distribution.items()
            ]
        )
        st.bar_chart(strat_df.set_index("Strategy")["Chunks"], height=220, color="#63b3ed")

    # ── Document table ──────────────────────────────────────────────
    if analytics.documents:
        st.divider()
        st.markdown("### Document Details")
        rows = []
        for doc in analytics.documents:
            # FIX: Enum Deserialization / Cache Unpickling Crash - safely resolve enum or str value
            status_val = getattr(doc.status, "value", str(doc.status)) if doc.status else "—"
            strat_val = getattr(doc.chunking_strategy, "value", str(doc.chunking_strategy)) if doc.chunking_strategy else "—"
            rows.append(
                {
                    "Document": doc.file_name,
                    "Source ID": doc.source,
                    "Status": status_val,
                    "Chunks": doc.chunk_count or 0,
                    "Strategy": strat_val,
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
