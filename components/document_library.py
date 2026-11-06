"""
components/document_library.py — Document upload, management, and status tracking.
"""

from __future__ import annotations
import asyncio
import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st

from custom_types import ChunkingStrategy, IngestionStatus


# ─────────────────────────────────────────────────────────────────────────────
# Helpers & Threaded Async Event Runner
# ─────────────────────────────────────────────────────────────────────────────


# FIX: Streamlit Threaded Async Event Loop Collision - Safe async runner to prevent RuntimeError in Streamlit thread
def _safe_run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(lambda: asyncio.run(coro)).result()
        return loop.run_until_complete(coro)


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def _fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _poll_ingestion_status(
    event_id: str, status_placeholder, timeout_s: float = 300.0
) -> tuple[IngestionStatus, dict]:
    """
    Poll Inngest for ingestion completion.
    Updates status_placeholder with live status.
    Returns (final_status, output_dict).
    """
    start = time.time()
    poll_interval = 1.0

    while True:
        try:
            runs = _fetch_runs(event_id)
            if runs:
                run = runs[0]
                raw_status = run.get("status", "")
                elapsed = time.time() - start

                if raw_status in ("Completed", "Succeeded", "Success", "Finished"):
                    output = run.get("output") or {}
                    ingested = output.get("ingested", 0)
                    if ingested <= 0:
                        time.sleep(1.0)
                        try:
                            updated_runs = _fetch_runs(event_id)
                            if updated_runs and updated_runs[0].get("output"):
                                output = updated_runs[0].get("output") or {}
                                ingested = output.get("ingested", 0)
                        except Exception:
                            pass
                    status_placeholder.success(
                        f"✅ **Indexed** — {ingested} chunks stored ({elapsed:.0f}s)"
                    )
                    return IngestionStatus.INDEXED, output

                elif raw_status in ("Failed", "Cancelled"):
                    status_placeholder.error(f"❌ Ingestion **{raw_status}** after {elapsed:.0f}s")
                    return IngestionStatus.FAILED, {}

                else:
                    status_placeholder.info(
                        f"⚙️ **Processing…** ({raw_status or 'Running'}) — {elapsed:.0f}s elapsed"
                    )
            else:
                elapsed = time.time() - start
                status_placeholder.info(f"📤 **Queued…** — {elapsed:.0f}s elapsed")

        except Exception as exc:
            status_placeholder.warning(f"⚠️ Status check error: {exc}")

        if time.time() - start > timeout_s:
            status_placeholder.error(f"⏱ Ingestion timed out after {timeout_s}s")
            return IngestionStatus.FAILED, {}

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.3, 5.0)


# FIX: Inngest App ID Decoupling - Align app_id to 'Prod_RAG_App' matching FastAPI backend worker
@st.cache_resource
def _get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="Prod_RAG_App", is_production=False)


def _save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    # Safe filename — avoid path traversal
    safe_name = Path(file.name).name
    file_path = uploads_dir / safe_name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def _send_ingest_event(pdf_path: Path, source_id: str, strategy: str) -> str:
    client = _get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": source_id,
                "strategy": strategy,
            },
        )
    )
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# Status badge helper
# ─────────────────────────────────────────────────────────────────────────────


def _status_badge(status: IngestionStatus) -> str:
    mapping = {
        IngestionStatus.UPLOADED:   ('<span class="badge badge-muted">Uploaded</span>', ""),
        IngestionStatus.QUEUED:     ('<span class="badge badge-warning">Queued</span>', ""),
        IngestionStatus.PROCESSING: ('<span class="badge badge-warning">Processing</span>', "⚙️"),
        IngestionStatus.INDEXED:    ('<span class="badge badge-success">Indexed</span>', "✅"),
        IngestionStatus.FAILED:     ('<span class="badge badge-error">Failed</span>', "❌"),
    }
    html, _ = mapping.get(status, ('<span class="badge badge-muted">Unknown</span>', ""))
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Document table
# ─────────────────────────────────────────────────────────────────────────────


def _render_document_table(analytics):
    """Render a table of indexed documents with actions."""
    if not analytics or not analytics.documents:
        st.info("No documents indexed yet. Upload a PDF below.")
        return

    for doc in analytics.documents:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(
                    f"**{doc.file_name}**  "
                    f"{_status_badge(doc.status)}",
                    unsafe_allow_html=True,
                )
                if doc.chunking_strategy:
                    strat_label = getattr(doc.chunking_strategy, "value", str(doc.chunking_strategy))
                    st.caption(f"Strategy: `{strat_label}`")
            with col2:
                st.metric("Chunks", doc.chunk_count or "—")
            with col3:
                if st.button("🔍 Explore", key=f"explore_{doc.source}", use_container_width=True):
                    st.session_state.explorer_source = doc.source
                    st.session_state.active_page = "explorer"
                    st.rerun()
            with col4:
                if st.button("🔄 Reprocess", key=f"reprocess_{doc.source}", use_container_width=True):
                    st.session_state.reprocess_source = doc.source
                    st.session_state.reprocess_path = doc.file_name
        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────


def render_library(analytics, refresh_analytics_fn):
    st.markdown("## 📚 Document Library")

    # ── Document table ──────────────────────────────────────────────
    st.markdown("### Indexed Documents")
    _render_document_table(analytics)

    if st.button("🔄 Refresh", key="refresh_library"):
        refresh_analytics_fn()
        st.rerun()

    st.divider()

    # ── Upload section ──────────────────────────────────────────────
    st.markdown("### Upload New Documents")

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="lib_uploader",
    )

    # FIX: Settings State Synchronization Across Views - Wire default strategy to settings_strategy
    default_strategy_val = st.session_state.get(
        "settings_strategy", ChunkingStrategy.HYBRID_SEMANTIC.value
    )
    strategy_options = [s.value for s in ChunkingStrategy]
    default_idx = (
        strategy_options.index(default_strategy_val)
        if default_strategy_val in strategy_options
        else 0
    )

    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox(
            "Chunking Strategy",
            options=strategy_options,
            index=default_idx,
            key="lib_strategy",
        )
    with col2:
        custom_source_prefix = st.text_input(
            "Source ID prefix (optional)",
            placeholder="e.g. project_a/",
            key="lib_source_prefix",
        )

    if uploaded_files:
        if st.button("📤 Upload & Index", type="primary", key="lib_upload_btn"):
            for uploaded in uploaded_files:
                safe_name = Path(uploaded.name).name
                source_id = (custom_source_prefix or "") + safe_name

                st.markdown(f"**Processing:** `{safe_name}`")
                progress_ph = st.empty()
                status_ph = st.empty()

                try:
                    # Save
                    progress_ph.info("💾 Saving file…")
                    path = _save_uploaded_pdf(uploaded)
                    progress_ph.success(f"✅ Saved: `{path.name}`")

                    # Queue
                    status_ph.info("📤 Sending to Inngest…")
                    # FIX: Streamlit Threaded Async Event Loop Collision - use _safe_run_async instead of asyncio.run
                    event_id = _safe_run_async(
                        _send_ingest_event(path, source_id, strategy)
                    )
                    status_ph.info("⚙️ Queued — waiting for ingestion to complete…")

                    # Poll
                    final_status, output = _poll_ingestion_status(event_id, status_ph)

                    if final_status == IngestionStatus.INDEXED:
                        ingested = output.get("ingested", 0)
                        if ingested <= 0:
                            try:
                                from vector_db import QdrantStorage
                                chunks = QdrantStorage().get_chunks_for_source(source_id)
                                if chunks:
                                    ingested = len(chunks)
                                    output["ingested"] = ingested
                            except Exception:
                                pass
                        st.success(
                            f"✅ **{safe_name}** indexed with **{ingested} chunks** "
                            f"using `{strategy}`."
                        )
                        refresh_analytics_fn()
                    else:
                        st.error(f"❌ Ingestion failed for `{safe_name}`.")

                except Exception as exc:
                    st.error(f"❌ Error processing `{safe_name}`: {exc}")

    # ── Reprocess pending ───────────────────────────────────────────
    reprocess_source = st.session_state.pop("reprocess_source", None)
    if reprocess_source:
        reprocess_path = st.session_state.pop("reprocess_path", reprocess_source)
        # Try to find the file in uploads/
        candidate = Path("uploads") / Path(reprocess_path).name
        if candidate.exists():
            st.markdown(f"**Reprocessing:** `{reprocess_source}`")
            status_ph = st.empty()
            try:
                # FIX: Settings State Synchronization - use configured default strategy
                reprocess_strat = st.session_state.get(
                    "settings_strategy", ChunkingStrategy.HYBRID_SEMANTIC.value
                )
                # FIX: Streamlit Threaded Async Event Loop Collision - use _safe_run_async
                event_id = _safe_run_async(
                    _send_ingest_event(
                        candidate, reprocess_source, reprocess_strat
                    )
                )
                _poll_ingestion_status(event_id, status_ph)
                refresh_analytics_fn()
            except Exception as exc:
                st.error(f"Reprocess failed: {exc}")
        else:
            st.warning(f"Original file not found at `{candidate}`. Upload it again to reprocess.")
