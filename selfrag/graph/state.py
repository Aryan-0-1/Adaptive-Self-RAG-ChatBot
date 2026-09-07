"""
state.py
--------
The LangGraph `State` TypedDict, plus the small Pydantic schemas used
for structured-output LLM calls throughout graph/nodes.py.
"""

from __future__ import annotations

from typing import List, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.documents import Document


class State(TypedDict):
    question: str

    # What we actually send to the local vector retriever (may differ
    # from `question` after a rewrite_question round).
    retrieval_query: str
    rewrite_tries: int

    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str

    # Post-generation verification (IsSUP) + revise loop
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int

    # Post-generation usefulness check (IsUSE)
    isuse: Literal["useful", "not_useful"]
    use_reason: str

    # Web-search fallback
    web_query: str
    web_search_used: bool

    # Where the final answer's context came from — surfaced in the UI.
    answer_source: Literal["general", "local", "web", "none"]


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ..., description="True if external documents are needed to answer reliably, else False."
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ..., description="True ONLY if the document contains info that can directly answer the question."
    )


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ..., description="Rewritten query optimized for vector retrieval against loaded sources."
    )


class WebQueryDecision(BaseModel):
    query: str = Field(..., description="Short keyword query optimized for a web search engine.")
