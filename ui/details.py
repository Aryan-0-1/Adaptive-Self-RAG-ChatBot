"""
details.py
----------
Renders the "🔍 View Self-RAG Details" expander under an answer:
retrieval decision, answer source (general/local/web), IsSUP/IsUSE
verdicts, evidence, retrieval query used, and the raw retrieved
excerpts / full context.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

_SOURCE_LABELS = {
    "general": "💡 General knowledge (no retrieval)",
    "local": "📄 Your loaded documents",
    "web": "🌐 Web search",
    "none": "🚫 No answer found",
}


def render_self_rag_details(result: Dict[str, Any]) -> None:
    need_retrieval = result.get("need_retrieval", False)
    answer_source = result.get("answer_source", "none")

    st.write("**Retrieval:**", "Required" if need_retrieval else "Not required")
    st.write("**Answer source:**", _SOURCE_LABELS.get(answer_source, answer_source))

    if not need_retrieval:
        st.caption(
            "This question was answered directly from general knowledge, so the "
            "retrieval / IsSUP / IsUSE steps never ran."
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        st.write("**IsSUP:**", result.get("issup", "—"))
    with c2:
        st.write("**IsUSE:**", result.get("isuse", "—"))

    retrieval_query = result.get("retrieval_query")
    if retrieval_query:
        st.write("**Local retrieval query**")
        st.code(str(retrieval_query))

    web_query = result.get("web_query")
    if answer_source == "web" and web_query:
        st.write("**Web search query**")
        st.code(str(web_query))

    if result.get("retries"):
        st.caption(f"🔄 Answer revised {result['retries']} time(s) for groundedness.")
    if result.get("rewrite_tries"):
        st.caption(f"🔁 Question rewritten & re-retrieved {result['rewrite_tries']} time(s).")

    evidence = result.get("evidence") or []
    if evidence:
        st.write("**Evidence**")
        for item in evidence:
            st.markdown(f"> {item}")

    use_reason = result.get("use_reason")
    if use_reason:
        st.write(f"**Usefulness reason:** {use_reason}")

    relevant_docs = result.get("relevant_docs") or []
    if relevant_docs:
        st.write("**Relevant source excerpts**")
        for i, doc in enumerate(relevant_docs, start=1):
            # `doc` is a live `Document` right after a graph run, or a
            # plain dict (`{"page_content": ..., "metadata": ...}`) once
            # it's been round-tripped through chat_store's JSON storage.
            if isinstance(doc, dict):
                content = doc.get("page_content", "")
                metadata = doc.get("metadata") or {}
            else:
                content = getattr(doc, "page_content", str(doc))
                metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source", "Unknown source")
            with st.expander(f"Source {i}: {source}"):
                st.write(content)

    context = result.get("context")
    if context:
        with st.expander("📚 Full retrieval context"):
            st.text(context)
