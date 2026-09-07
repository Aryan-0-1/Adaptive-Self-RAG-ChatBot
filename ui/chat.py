"""
chat.py
-------
Session-state initialization, thread switching (backed by
chat_store.py + a re-embed into the global FAISS store), chat history
rendering, and the input box that drives a Self-RAG run.
"""

from __future__ import annotations

import traceback
import uuid

import streamlit as st

from selfrag import chat_store
from ui.details import render_self_rag_details
from ui.sidebar import _resync_vectorstore_for_thread
from ui.streaming import describe_step, run_self_rag_streaming


def init_session_state() -> None:
    chat_store.init_db()

    if "thread_id" not in st.session_state:
        thread_id = str(uuid.uuid4())
        chat_store.create_thread(thread_id)
        st.session_state["thread_id"] = thread_id
        st.session_state["source_names"] = []
        st.session_state["youtube_urls"] = []
        st.session_state["_synced_thread_id"] = None

    # Re-embed this thread's sources into the (process-global) FAISS
    # index if we haven't already done so in this session/rerun cycle —
    # e.g. right after switching threads, or on a fresh process start.
    if st.session_state.get("_synced_thread_id") != st.session_state["thread_id"]:
        _resync_vectorstore_for_thread(st.session_state["thread_id"])
        st.session_state["_synced_thread_id"] = st.session_state["thread_id"]


def start_new_thread() -> None:
    thread_id = str(uuid.uuid4())
    chat_store.create_thread(thread_id)
    st.session_state["thread_id"] = thread_id
    st.session_state["source_names"] = []
    st.session_state["youtube_urls"] = []
    st.session_state["_synced_thread_id"] = None


def switch_thread(thread_id: str) -> None:
    st.session_state["thread_id"] = thread_id
    st.session_state["_synced_thread_id"] = None


def render_history() -> None:
    for msg in chat_store.get_messages(st.session_state["thread_id"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("metadata"):
                with st.expander("🔍 View Self-RAG Details"):
                    render_self_rag_details(msg["metadata"])


def handle_user_input() -> None:
    user_input = st.chat_input("Ask a question about your documents or video...")
    if not user_input or not user_input.strip():
        return

    user_input = user_input.strip()
    thread_id = st.session_state["thread_id"]

    chat_store.add_message(thread_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status = st.status("🧠 Running Self-RAG...", expanded=True)
        result = {}
        last_node = None
        stream = run_self_rag_streaming(user_input)
        try:
            for node_name, node_output in stream:
                last_node = node_name
                # A node returning None instead of a dict would otherwise
                # crash here with "'NoneType' object is not iterable" from
                # dict.update(None), which masks whatever actually went
                # wrong inside that node. Skip the merge instead so the
                # loop can keep going (or hit a clearer error downstream),
                # and surface which node did it via `last_node` below.
                if node_output:
                    result.update(node_output)
                status.write(describe_step(node_name))

            answer = str(result.get("answer") or "No answer found.")
            status.update(label="✅ Answer generated", state="complete", expanded=False)
            st.markdown(answer)

            with st.expander("🔍 View Self-RAG Details"):
                render_self_rag_details(result)

            chat_store.add_message(thread_id, "assistant", answer, metadata=result)
            chat_store.touch_thread(thread_id, title=user_input[:60])

        except Exception as exc:
            status.update(label="❌ Error", state="error", expanded=True)
            if last_node:
                st.error(f"Error after node `{last_node}`: {exc}")
            else:
                st.error(str(exc))
            with st.expander("🪲 Full traceback (for debugging)"):
                st.code(traceback.format_exc())
        finally:
            # If we broke out of the loop early (exception above, or the
            # Streamlit script itself gets interrupted mid-run), close the
            # generator here rather than leaving it for garbage collection —
            # see run_self_rag_streaming()'s docstring for why that matters.
            stream.close()
