"""
config.py
---------
Central place for environment configuration and constants.

Keys can arrive two different ways depending on where the app runs:

- Locally: a `.env` file read by `python-dotenv`.
- On Streamlit Community Cloud: `st.secrets`, populated from the app's
  "Secrets" dashboard setting (no `.env` file exists on Cloud at all).

`get_env()` checks `st.secrets` first, then falls back to `os.environ`,
so the exact same code works in both places. Every module that needs a
key or tunable should read it through `get_env()` rather than calling
`os.getenv()` directly.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

# No-op if no .env file exists (e.g. on Streamlit Cloud), so this is
# always safe to call.
load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a config value from st.secrets, then os.environ, then default."""
    try:
        import streamlit as st

        if key in st.secrets:
            value = st.secrets[key]
            if value not in (None, ""):
                return str(value)
    except Exception:
        # st.secrets raises if no secrets.toml exists (e.g. local dev
        # without Streamlit Cloud) and Streamlit may not even be
        # importable in non-UI contexts — either way, fall through.
        pass

    return os.getenv(key, default)


# ---------------------------------------------------------------
# Chat LLM (Ollama local or Ollama Cloud)
# ---------------------------------------------------------------
OLLAMA_API_KEY = get_env("OLLAMA_API_KEY")
OLLAMA_CHAT_MODEL = get_env("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
OLLAMA_LOCAL_HOST = get_env("OLLAMA_LOCAL_HOST", "http://localhost:11434")
OLLAMA_HOST = get_env("OLLAMA_HOST", "https://ollama.com")

# ---------------------------------------------------------------
# Embeddings (Google Generative AI)
# ---------------------------------------------------------------
GOOGLE_API_KEY = get_env("GOOGLE_API_KEY")
GOOGLE_EMBED_MODEL = get_env("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")
GOOGLE_EMBED_BATCH_SIZE = int(get_env("GOOGLE_EMBED_BATCH_SIZE", "50"))
GOOGLE_EMBED_MAX_RETRIES = int(get_env("GOOGLE_EMBED_MAX_RETRIES", "3"))

# ---------------------------------------------------------------
# Web-search fallback (Tavily, optional)
# ---------------------------------------------------------------
TAVILY_API_KEY = get_env("TAVILY_API_KEY")

# ---------------------------------------------------------------
# Chat persistence
# ---------------------------------------------------------------
CHAT_DB_PATH = get_env("CHAT_DB_PATH", "self_rag_chat_history.sqlite3")

# ---------------------------------------------------------------
# LangSmith tracing (optional; langchain reads these from the
# environment itself, we just make sure they're exported)
# ---------------------------------------------------------------
for _key in (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
):
    _val = get_env(_key)
    if _val is not None:
        os.environ[_key] = _val

# ---------------------------------------------------------------
# Feature-readiness flags — used by the UI to show banners instead
# of crashing when optional keys are missing.
# ---------------------------------------------------------------
EMBEDDINGS_READY = bool(GOOGLE_API_KEY)
WEB_SEARCH_READY = bool(TAVILY_API_KEY)


def embeddings_ready() -> bool:
    return EMBEDDINGS_READY


def web_search_ready() -> bool:
    return WEB_SEARCH_READY

# ---------------------------------------------------------------
# Graph tuning constants
# ---------------------------------------------------------------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
RETRIEVER_TOP_K = 4

MAX_SUP_RETRIES = 2          # revise_answer <-> is_sup loop cap
MAX_REWRITE_TRIES = 2        # local rewrite_question <-> retrieve loop cap

# Worst case per query (see graph/build.py docstring for the full trace):
# decide_retrieval(1) + [retrieve+is_relevant](1) each local rewrite round
# (rewrite_question+retrieve+is_relevant = 3) x MAX_REWRITE_TRIES(2) = 6
# + web fallback round (rewrite_query_for_web+web_search+is_relevant = 3)
# + generate_from_context+is_sup pair, revised up to MAX_SUP_RETRIES(2)
# times (is_sup+revise_answer = 2) x 2 = 4, plus final is_sup = 1
# + is_use, possibly repeated once per rewrite/web round (3 rounds) = 3
# Sum ≈ 1 + 2 + 6 + 3 + 1 + 4 + 1 + 3 ≈ 21 nodes for one full worst-case
# question. We set recursion_limit well above that for headroom.
GRAPH_RECURSION_LIMIT = 80
