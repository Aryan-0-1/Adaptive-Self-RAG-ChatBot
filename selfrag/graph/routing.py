"""
routing.py
----------
All `route_after_*` functions used as LangGraph conditional edges.

See graph/build.py's module docstring for the full reasoning behind
the web-fallback ordering/guards implemented here. Short version:

- Local `rewrite_question -> retrieve` retries are tried before the
  web fallback, because they're free (no external API call) and reuse
  documents already indexed. Web search is a paid/rate-limited external
  call plus another full generate+verify pass, so it's reserved for
  when local retries are exhausted (or local retrieval found nothing
  to rewrite towards in the first place).
- `web_search_used` is a single-use guard (not a counter) — we only
  ever call Tavily once per question. `rewrite_tries` is a separate,
  independent counter — so the two budgets can't interact to produce
  an infinite loop; each is checked and incremented independently,
  and once both are exhausted every path in `route_after_relevance`
  and `route_after_isuse` leads to `no_answer_found`.
- A web-sourced answer that fails IsSUP/IsUSE IS allowed to fall back
  to local retrieval again (if local retries aren't exhausted yet) —
  local retries are cheap, so there's no reason to give up on them
  just because the web path already ran once.
"""

from __future__ import annotations

from typing import Literal

from .. import config
from .state import State


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def route_after_relevance(state: State) -> Literal["generate_from_context", "web_fallback", "no_answer_found"]:
    if state.get("relevant_docs"):
        return "generate_from_context"

    if config.WEB_SEARCH_READY and not state.get("web_search_used"):
        return "web_fallback"

    return "no_answer_found"


def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"

    if state.get("retries", 0) >= config.MAX_SUP_RETRIES:
        return "accept_answer"  # give up revising; let IsUSE have the final word

    return "revise_answer"


def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "web_fallback", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"

    if state.get("rewrite_tries", 0) < config.MAX_REWRITE_TRIES:
        return "rewrite_question"

    if config.WEB_SEARCH_READY and not state.get("web_search_used"):
        return "web_fallback"

    return "no_answer_found"
