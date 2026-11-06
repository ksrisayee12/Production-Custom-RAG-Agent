"""
components/chat_view.py — Premium chat interface with source citations.
"""

from __future__ import annotations
import asyncio
import os
import time

import requests
import streamlit as st
import inngest

from custom_types import RetrievedChunk


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────


def _init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_top_k" not in st.session_state:
        # FIX: Settings State Synchronization Across Views - Wire top-k default to settings_top_k
        st.session_state.query_top_k = st.session_state.get("settings_top_k", 5)
    if "query_source_filter" not in st.session_state:
        st.session_state.query_source_filter = None
    if "enable_neighbors" not in st.session_state:
        # FIX: Settings State Synchronization Across Views - Wire neighbor expansion default to settings_enable_neighbors
        st.session_state.enable_neighbors = st.session_state.get("settings_enable_neighbors", True)


# ─────────────────────────────────────────────────────────────────────────────
# Inngest query helpers & Threaded Async Event Runner
# ─────────────────────────────────────────────────────────────────────────────


# FIX: Streamlit Threaded Async Event Loop Collision - Safe async runner to prevent RuntimeError
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


def _wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.8,
) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = _fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Inngest function run {status}")
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for run (last status: {last_status})"
            )
        time.sleep(poll_interval_s)


# FIX: Inngest App ID Decoupling - Align app_id to 'Prod_RAG_App' matching FastAPI backend worker
@st.cache_resource
def _get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="Prod_RAG_App", is_production=False)


async def _send_query_event(question: str, top_k: int, source_id, enable_neighbors: bool) -> str:
    client = _get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
                "source_id": source_id,
                "enable_neighbors": enable_neighbors,
            },
        )
    )
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────────────────────


def _render_welcome(available_sources: list[str]):
    st.markdown(
        """
        <div class="welcome-hero">
          <div class="welcome-title">Document Intelligence</div>
          <div class="welcome-subtitle">
            Ask questions about your indexed documents. Answers are grounded in
            retrieved passages with source and page attribution.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if available_sources:
        suggestions = [
            "What are the main topics covered?",
            "Summarize the key findings.",
            "What conclusions are drawn?",
            "List the main recommendations.",
        ]
        st.markdown("**Suggested questions:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()
    else:
        st.info("📂 Upload and index a PDF in **Document Library** to get started.", icon="💡")


# ─────────────────────────────────────────────────────────────────────────────
# Message rendering
# ─────────────────────────────────────────────────────────────────────────────


def _render_source_citations(chunks: list[dict]):
    """Render inline source pills for citations."""
    if not chunks:
        return

    with st.expander("📎 Retrieved Sources", expanded=False):
        for i, chunk in enumerate(chunks[:8]):
            source = chunk.get("source", "")
            page = chunk.get("page_number")
            score = chunk.get("score", 0.0)
            section = chunk.get("section", "")
            text = chunk.get("text", "")[:300]
            is_neighbor = chunk.get("is_neighbor", False)

            label_parts = [f"**{source}**"]
            if page:
                label_parts.append(f"p.{page}")
            if section:
                label_parts.append(f"§ {section}")
            if is_neighbor:
                label_parts.append("*(context)*")

            st.markdown(
                f"<span class='badge badge-info'>#{i+1}</span> "
                f"{' · '.join(label_parts)} "
                f"<span class='badge badge-muted'>score: {score:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"_{text}..._")
            if i < len(chunks) - 1:
                st.divider()


def _render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    chunks = msg.get("retrieved_chunks", [])

    if role == "user":
        st.markdown(
            f"""
            <div class='msg-user'>
              <div class='msg-role user'>You</div>
              {content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='msg-assistant'>
              <div class='msg-role assistant'>🧠 DocIntel</div>
              {content}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if chunks:
            _render_source_citations(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Main chat view
# ─────────────────────────────────────────────────────────────────────────────


def render_chat(available_sources: list[str]):
    _init_chat_state()

    # ── Settings bar ───────────────────────────────────────────────
    with st.expander("⚙️ Query Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.query_top_k = st.number_input(
                "Top-K chunks", min_value=1, max_value=20,
                value=st.session_state.query_top_k, step=1,
            )
        with col2:
            source_options = ["All Documents"] + available_sources
            selected = st.selectbox("Filter by document", source_options)
            st.session_state.query_source_filter = (
                None if selected == "All Documents" else selected
            )
        with col3:
            st.session_state.enable_neighbors = st.toggle(
                "Neighbor expansion", value=st.session_state.enable_neighbors
            )

    st.divider()

    # ── Messages area ───────────────────────────────────────────────
    if not st.session_state.messages:
        _render_welcome(available_sources)
    else:
        for msg in st.session_state.messages:
            _render_message(msg)

    st.divider()

    # ── Handle suggestion click ─────────────────────────────────────
    pending = st.session_state.pop("pending_question", None)

    # ── Input area ──────────────────────────────────────────────────
    col_form, col_clear = st.columns([7, 1])
    with col_form:
        with st.form("chat_form", clear_on_submit=True):
            f_col_input, f_col_btn = st.columns([6, 1])
            with f_col_input:
                question = st.text_input(
                    "Ask a question",
                    value=pending or "",
                    placeholder="Type your question here… (Press Enter to send)",
                    label_visibility="collapsed",
                    key="chat_input_field",
                )
            with f_col_btn:
                send = st.form_submit_button("Send", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Process query ───────────────────────────────────────────────
    if (send or pending) and (question or pending):
        q = (pending or question).strip()
        if not q:
            return

        st.session_state.messages.append({"role": "user", "content": q})

        with st.spinner("🔍 Searching documents and generating answer…"):
            try:
                # FIX: Streamlit Threaded Async Event Loop Collision - use _safe_run_async instead of asyncio.run
                event_id = _safe_run_async(
                    _send_query_event(
                        q,
                        st.session_state.query_top_k,
                        st.session_state.query_source_filter,
                        st.session_state.enable_neighbors,
                    )
                )
                output = _wait_for_run_output(event_id)
                answer = output.get("answer", "(No answer returned)")
                retrieved_chunks = output.get("retrieved_chunks", [])
                sources = output.get("sources", [])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "retrieved_chunks": retrieved_chunks,
                        "sources": sources,
                    }
                )
            except TimeoutError:
                st.error("⏱ Query timed out. Please try again.")
                st.session_state.messages.pop()
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                st.session_state.messages.pop()

        st.rerun()
