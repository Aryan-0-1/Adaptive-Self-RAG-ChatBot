"""
build.py
--------
Builds and compiles the Self-RAG StateGraph. Exposes the compiled
graph as `app`, imported by the UI layer (`ui/streaming.py`) and by
`frontend_self_rag.py`.

Full flow::

    START -> decide_retrieval ─┬─▶ generate_direct ─────────────────────────────▶ END
                                └─▶ retrieve (local) -> is_relevant ─┬─▶ generate_from_context -> is_sup ⇄ revise_answer -> is_use ─┬─▶ END
                                                                     │                                                             ├─▶ rewrite_question -> retrieve (retry)
                                                                     └─▶ rewrite_query_for_web -> web_search -> is_relevant ────────┴─▶ no_answer_found -> END
                                                                          (web_fallback; only if no local docs were relevant,
                                                                           or after local retries + IsUSE still say not_useful)

`is_relevant` is reused for both local and web-search results — see
graph/nodes.py's `web_search()` docstring for why a separate
`is_relevant_web` node isn't needed. Its two possible destinations
(`generate_from_context` or, if still nothing relevant, back into the
web fallback / `no_answer_found`) are both handled by the single
`route_after_relevance` function in routing.py.

Worst-case node count for one query (used to size `recursion_limit`
in config.py): 1 (decide) + [retrieve+is_relevant] + up to
MAX_REWRITE_TRIES rounds of [rewrite_question+retrieve+is_relevant]
+ one web-fallback round of [rewrite_query_for_web+web_search+
is_relevant] + up to MAX_SUP_RETRIES rounds of [is_sup+revise_answer]
(possibly repeated once per local/web round) + is_use per round.
With the defaults (MAX_REWRITE_TRIES=2, MAX_SUP_RETRIES=2) this comes
to roughly 20-25 node visits; `GRAPH_RECURSION_LIMIT=80` leaves
comfortable headroom.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .routing import route_after_decide, route_after_issup, route_after_isuse, route_after_relevance
from .state import State


def build_graph():
    g = StateGraph(State)

    g.add_node("decide_retrieval", nodes.decide_retrieval)
    g.add_node("generate_direct", nodes.generate_direct)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("is_relevant", nodes.is_relevant)
    g.add_node("generate_from_context", nodes.generate_from_context)
    g.add_node("no_answer_found", nodes.no_answer_found)

    g.add_node("is_sup", nodes.is_sup)
    g.add_node("revise_answer", nodes.revise_answer)
    g.add_node("accept_answer", nodes.accept_answer)

    g.add_node("is_use", nodes.is_use)
    g.add_node("rewrite_question", nodes.rewrite_question)

    g.add_node("rewrite_query_for_web", nodes.rewrite_query_for_web)
    g.add_node("web_search", nodes.web_search)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )
    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "web_fallback": "rewrite_query_for_web",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("no_answer_found", END)

    g.add_edge("generate_from_context", "is_sup")
    g.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {"accept_answer": "accept_answer", "revise_answer": "revise_answer"},
    )
    g.add_edge("revise_answer", "is_sup")
    g.add_edge("accept_answer", "is_use")

    g.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {
            "END": END,
            "rewrite_question": "rewrite_question",
            "web_fallback": "rewrite_query_for_web",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("rewrite_question", "retrieve")

    g.add_edge("rewrite_query_for_web", "web_search")
    g.add_edge("web_search", "is_relevant")

    return g.compile()


app = build_graph()
