"""
streaming.py
------------
Runs the compiled graph via `app.stream(..., stream_mode="updates")` so
the UI can show a live, node-by-node progress trail instead of a single
blocking `invoke()` call. `describe_step()` maps a node name to a short
human-readable label — if you add/rename/remove a graph node, update
the mapping here too, or its progress line will silently go missing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

from selfrag import config
from selfrag.graph import app

_STEP_LABELS = {
    "decide_retrieval": "🧭 Deciding whether retrieval is needed…",
    "generate_direct": "💡 Answering from general knowledge…",
    "retrieve": "🔎 Retrieving from your loaded sources…",
    "is_relevant": "🧩 Filtering relevant documents…",
    "generate_from_context": "✍️ Generating a grounded answer…",
    "is_sup": "🛡️ Checking groundedness (IsSUP)…",
    "revise_answer": "🔄 Revising the answer to stay grounded…",
    "accept_answer": "✅ Answer accepted as grounded…",
    "is_use": "🎯 Checking answer usefulness (IsUSE)…",
    "rewrite_question": "🔁 Rewriting the query and retrying retrieval…",
    "rewrite_query_for_web": "🌐 Preparing a web search query…",
    "web_search": "🌐 Searching the web…",
    "no_answer_found": "🚫 No supported answer could be found…",
}


def describe_step(node_name: str) -> str:
    return _STEP_LABELS.get(node_name, f"➡️ {node_name}…")


def initial_state(question: str) -> Dict[str, Any]:
    return {
        "question": question,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "need_retrieval": False,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "no_support",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
        "web_query": "",
        "web_search_used": False,
        "answer_source": "none",
    }


def run_self_rag_streaming(question: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yields (node_name, node_output) pairs as the graph executes.

    If the caller stops iterating early (an exception in the loop body,
    or Streamlit interrupting the script mid-run for a rerun/hot-reload),
    Python closes this generator via GeneratorExit. We explicitly close
    the inner `app.stream()` iterator ourselves in `finally` so that
    cleanup happens deterministically here rather than whenever the
    garbage collector gets to it — LangGraph's own generator doesn't
    always swallow GeneratorExit silently, which is what produces the
    "GeneratorExit / yield from _output(...)" traceback in the terminal
    if this isn't done explicitly.
    """
    inputs = initial_state(question)
    stream_iter = app.stream(
        inputs, config={"recursion_limit": config.GRAPH_RECURSION_LIMIT}, stream_mode="updates"
    )
    try:
        for update in stream_iter:
            for node_name, node_output in update.items():
                yield node_name, node_output
    finally:
        stream_iter.close()


def run_self_rag(question: str) -> Dict[str, Any]:
    """Non-streaming convenience wrapper — runs the graph to completion
    and returns the final merged state."""
    final_state: Dict[str, Any] = dict(initial_state(question))
    for _, node_output in run_self_rag_streaming(question):
        final_state.update(node_output)
    return final_state
