"""
nodes.py
--------
All LangGraph node functions for the Self-RAG workflow:

  decide_retrieval → retrieve → is_relevant ─┬─▶ generate_from_context → is_sup ⇄ revise_answer → is_use
                                              │                                                       │
                                              └─▶ rewrite_query_for_web → web_search ──────────────────┘
                                       (is_relevant is reused for both local and web docs — see routing.py)

Routing logic (which node runs next) lives in routing.py, not here —
this file only contains the node bodies themselves.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .. import config
from ..llm import llm
from ..vectorstore import get_retriever
from .state import (
    IsSUPDecision,
    IsUSEDecision,
    RelevanceDecision,
    RetrieveDecision,
    RewriteDecision,
    State,
    WebQueryDecision,
)

# =================================================================
# 1) Decide retrieval
# =================================================================
_decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from the loaded PDFs or video transcripts.\n"
            "- should_retrieve=False only for general knowledge/definitions unrelated to any loaded source.\n"
            "- If unsure, choose True.",
        ),
        ("human", "Question: {question}"),
    ]
)
_should_retrieve_llm = llm.with_structured_output(RetrieveDecision)


def decide_retrieval(state: State) -> dict:
    decision: RetrieveDecision = _should_retrieve_llm.invoke(
        _decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}


# =================================================================
# 2) Direct answer (no retrieval)
# =================================================================
_direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only your general knowledge.\n"
            "If it requires specific source info, say:\n"
            "'I don't know based on my general knowledge.'",
        ),
        ("human", "{question}"),
    ]
)


def generate_direct(state: State) -> dict:
    out = llm.invoke(_direct_generation_prompt.format_messages(question=state["question"]))
    return {"answer": out.content, "answer_source": "general"}


# =================================================================
# 3) Retrieve (local FAISS)
# =================================================================
def retrieve(state: State) -> dict:
    retriever = get_retriever()
    if retriever is None:
        return {"docs": []}
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}


# =================================================================
# 4) Relevance filter — shared by local docs AND web-search docs
# =================================================================
_is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)
_relevance_llm = llm.with_structured_output(RelevanceDecision)


def is_relevant(state: State) -> dict:
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision: RelevanceDecision = _relevance_llm.invoke(
            _is_relevant_prompt.format_messages(
                question=state["question"], document=doc.page_content
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


# =================================================================
# 5) Generate from context
# =================================================================
_rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a RAG chatbot.\n\n"
            "You will receive a CONTEXT block from retrieved documents.\n"
            "Task: answer the question using the context.\n"
            "Don't mention that you were given a context in your answer.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


def generate_from_context(state: State) -> dict:
    relevant_docs = state.get("relevant_docs", [])
    context = "\n\n---\n\n".join(d.page_content for d in relevant_docs).strip()
    if not context:
        return {"answer": "No answer found.", "context": "", "answer_source": "none"}

    out = llm.invoke(
        _rag_generation_prompt.format_messages(question=state["question"], context=context)
    )
    source = "web" if any(d.metadata.get("source") == "web" for d in relevant_docs) else "local"
    return {"answer": out.content, "context": context, "answer_source": source}


def no_answer_found(state: State) -> dict:
    return {"answer": "No answer found.", "context": "", "answer_source": "none"}


# =================================================================
# 6) IsSUP verify + revise loop
# =================================================================
_issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported: every meaningful claim is explicitly supported by CONTEXT,\n"
            "  and the ANSWER introduces no qualitative/interpretive words absent from CONTEXT.\n"
            "- partially_supported: core facts are supported, but the ANSWER adds any\n"
            "  abstraction/interpretation/qualitative phrasing not explicitly in CONTEXT.\n"
            "- no_support: the key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: any unsupported qualitative/interpretive phrasing -> partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported -> no_support.\n"
            "- Evidence: up to 3 short direct quotes from CONTEXT supporting the supported parts.\n"
            "- Do not use outside knowledge.",
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n",
        ),
    ]
)
_issup_llm = llm.with_structured_output(IsSUPDecision)


def is_sup(state: State) -> dict:
    decision: IsSUPDecision = _issup_llm.invoke(
        _issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    # Small local models (e.g. qwen2.5:3b) occasionally emit a literal
    # `null` for optional-looking fields instead of omitting them, which
    # bypasses Pydantic's default_factory. Coerce defensively rather than
    # let a downstream `dict.update(None)`-style crash mask this.
    return {"issup": decision.issup, "evidence": decision.evidence or []}


def accept_answer(state: State) -> dict:
    return {}  # keep answer as-is; this node exists purely for routing clarity


_revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "Output format (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n",
        ),
        (
            "human",
            "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}",
        ),
    ]
)


def revise_answer(state: State) -> dict:
    out = llm.invoke(
        _revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}


# =================================================================
# 7) IsUSE verify
# =================================================================
_isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: the answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: the answer is generic, off-topic, or only gives related background.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that) — only check 'did we answer the question?'\n"
            "- Keep reason to 1 short line.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]
)
_isuse_llm = llm.with_structured_output(IsUSEDecision)


def is_use(state: State) -> dict:
    decision: IsUSEDecision = _isuse_llm.invoke(
        _isuse_prompt.format_messages(question=state["question"], answer=state.get("answer", ""))
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason or ""}


# =================================================================
# 8) Local rewrite (retry local retrieval with a better query)
# =================================================================
_rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over the\n"
            "loaded PDFs/YouTube transcripts.\n\n"
            "Rules:\n"
            "- Keep it short (6-16 words).\n"
            "- Preserve key entities (names, plan names, etc.).\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query",
        ),
        (
            "human",
            "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer (if any):\n{answer}",
        ),
    ]
)
_rewrite_llm = llm.with_structured_output(RewriteDecision)


def rewrite_question(state: State) -> dict:
    decision: RewriteDecision = _rewrite_llm.invoke(
        _rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


# =================================================================
# 9) Web-search fallback (Tavily)
# =================================================================
_rewrite_for_web_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-14 words).\n"
            "- If the question implies recency, add '(last 30 days)'.\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)
_web_query_llm = llm.with_structured_output(WebQueryDecision)


def rewrite_query_for_web(state: State) -> dict:
    decision: WebQueryDecision = _web_query_llm.invoke(
        _rewrite_for_web_prompt.format_messages(question=state["question"])
    )
    return {"web_query": decision.query, "docs": [], "relevant_docs": [], "context": ""}


def _get_tavily_client():
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=5, tavily_api_key=config.TAVILY_API_KEY)


def web_search(state: State) -> dict:
    """Run a Tavily web search and wrap results as Documents tagged
    `metadata["source"] == "web"`, so the shared `is_relevant` /
    `generate_from_context` nodes downstream can tell where a chunk
    of context came from without needing web-specific copies of
    themselves (see routing.py + config.py's docstring for why we
    merged what would otherwise be an `is_relevant_web` node into the
    existing `is_relevant`)."""
    q = state.get("web_query") or state["question"]

    if not config.WEB_SEARCH_READY:
        return {"docs": [], "web_search_used": True}

    try:
        client = _get_tavily_client()
        response = client.invoke({"query": q})
        results = response.get("results", []) if isinstance(response, dict) else response
    except Exception:
        results = []

    docs: List[Document] = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        docs.append(Document(page_content=text, metadata={"source": "web", "url": url, "title": title}))

    return {"docs": docs, "web_search_used": True}
