"""
sidebar.py
----------
Sidebar: pipeline configuration, live workflow diagram, PDF/YouTube
source upload, current-sources summary, and the previous-chats list
(backed by chat_store.py).
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from selfrag import chat_store, config, vectorstore
from selfrag.graph import app as graph_app


def load_pdf(uploaded_file) -> List[Document]:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    try:
        docs = PyPDFLoader(temp_path).load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        return docs
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _extract_youtube_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
        r"(?:youtube\.com/embed/)([^?&]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


def load_youtube(url: str) -> List[Document]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError(
            "youtube-transcript-api is not installed. Run: pip install youtube-transcript-api"
        ) from exc

    video_id = _extract_youtube_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")

    api = YouTubeTranscriptApi()
    try:
        transcript = api.fetch(video_id)
    except Exception:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

    pieces = []
    for item in transcript:
        text = item.text if hasattr(item, "text") else item.get("text", "") if isinstance(item, dict) else str(item)
        if text:
            pieces.append(text)

    full_text = "\n".join(pieces).strip()
    if not full_text:
        raise ValueError("No transcript could be extracted from this video.")

    return [Document(page_content=full_text, metadata={"source": url, "type": "youtube", "video_id": video_id})]


def _add_source(thread_id: str, new_docs: List[Document], source_name: str) -> float:
    if not new_docs:
        return 0.0
    start = time.time()
    with st.spinner(f"Embedding '{source_name}' (Google embeddings, batched)…"):
        vectorstore.add_documents(new_docs)
    elapsed = time.time() - start
    chat_store.add_source(thread_id, source_name, new_docs)
    if source_name not in st.session_state["source_names"]:
        st.session_state["source_names"].append(source_name)
    return elapsed


def _clear_sources(thread_id: str) -> None:
    vectorstore.clear_documents()
    chat_store.clear_sources(thread_id)
    st.session_state["source_names"] = []
    st.session_state["youtube_urls"] = []


def _resync_vectorstore_for_thread(thread_id: str) -> None:
    """Documents live in chat_store per-thread, but the FAISS index is a
    single global store — re-embed the active thread's sources into it
    whenever we switch threads or the process restarts."""
    vectorstore.clear_documents()
    sources = chat_store.get_sources(thread_id)
    st.session_state["source_names"] = list(sources.keys())
    if not sources or not config.EMBEDDINGS_READY:
        return
    all_docs = [d for docs in sources.values() for d in docs]
    with st.spinner("Re-embedding this thread's sources…"):
        vectorstore.add_documents(all_docs)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Sources")

        if not config.EMBEDDINGS_READY:
            st.warning(
                "GOOGLE_API_KEY isn't configured — document upload/retrieval is "
                "disabled. You can still ask general-knowledge questions.",
                icon="⚠️",
            )
        if not config.WEB_SEARCH_READY:
            st.info(
                "TAVILY_API_KEY isn't configured — the web-search fallback is disabled.",
                icon="ℹ️",
            )

        if st.button("🆕 New Chat", use_container_width=True):
            from ui.chat import start_new_thread

            start_new_thread()
            st.rerun()

        with st.expander("🔧 Pipeline configuration"):
            for key, value in vectorstore.get_config_summary().items():
                st.write(f"**{key}:** {value}")

        with st.expander("🗺️ Workflow diagram"):
            try:
                st.image(graph_app.get_graph().draw_mermaid_png())
            except Exception:
                st.code(graph_app.get_graph().draw_mermaid(), language="mermaid")

        st.divider()
        thread_id = st.session_state["thread_id"]

        st.subheader("📄 PDF Documents")
        uploaded_pdfs = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True,
            disabled=not config.EMBEDDINGS_READY,
        )
        if st.button("➕ Add PDFs", use_container_width=True, disabled=not uploaded_pdfs):
            added, total_elapsed = 0, 0.0
            for uploaded_pdf in uploaded_pdfs:
                try:
                    docs = load_pdf(uploaded_pdf)
                    total_elapsed += _add_source(thread_id, docs, uploaded_pdf.name)
                    added += 1
                except Exception as exc:
                    st.error(f"Could not load `{uploaded_pdf.name}`: {exc}")
            if added:
                st.success(
                    f"Added {added} PDF source(s) in {total_elapsed:.1f}s — "
                    f"index now has {vectorstore.get_chunk_count()} chunk(s)."
                )

        st.subheader("▶️ YouTube")
        youtube_url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
            disabled=not config.EMBEDDINGS_READY,
        )
        if st.button("➕ Add YouTube", use_container_width=True, disabled=not youtube_url.strip()):
            try:
                url = youtube_url.strip()
                if url in st.session_state["youtube_urls"]:
                    st.warning("This YouTube video is already loaded.")
                else:
                    docs = load_youtube(url)
                    elapsed = _add_source(thread_id, docs, url)
                    st.session_state["youtube_urls"].append(url)
                    st.success(
                        f"YouTube transcript added in {elapsed:.1f}s — "
                        f"index now has {vectorstore.get_chunk_count()} chunk(s)."
                    )
            except Exception as exc:
                st.error(f"Could not load YouTube transcript: {exc}")

        st.divider()
        st.subheader("📚 Current Sources")
        if st.session_state["source_names"]:
            for source in st.session_state["source_names"]:
                st.caption(f"• {source}")
            st.caption(f"{vectorstore.get_chunk_count()} chunk(s) indexed.")
        else:
            st.info("No sources loaded yet.")

        if st.button(
            "🗑️ Clear Sources", use_container_width=True,
            disabled=not st.session_state["source_names"],
        ):
            _clear_sources(thread_id)
            st.rerun()

        st.divider()
        st.subheader("🗂️ Previous chats")
        for t in chat_store.list_threads():
            label = t["title"] or t["thread_id"][:8]
            if st.button(label, key=f"thread_{t['thread_id']}", use_container_width=True):
                from ui.chat import switch_thread

                switch_thread(t["thread_id"])
                st.rerun()
