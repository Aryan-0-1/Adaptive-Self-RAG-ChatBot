# Self-RAG Assistant

A LangGraph Self-RAG chatbot with a Streamlit UI: ask questions over PDFs or
YouTube transcripts, get answers that are retrieved, relevance-filtered,
grounding-checked (`IsSUP`), revised if needed, usefulness-checked (`IsUSE`),
and — if your local documents come up empty or a few local retries still
don't produce a useful answer — automatically backed by a live web search.

## Workflow

```
decide_retrieval ─┬─▶ generate_direct ──────────────────────────────────────────────▶ END
                   └─▶ retrieve (local) → is_relevant ─┬─▶ generate_from_context → is_sup ⇄ revise_answer → is_use ─┬─▶ END
                                                        │                                                          ├─▶ rewrite_question → retrieve (retry locally)
                                                        └─▶ rewrite_query_for_web → web_search → is_relevant ───────┴─▶ no_answer_found → END
                                                             (web fallback — only if local retrieval found nothing,
                                                              or after local rewrite retries are exhausted)
```

`is_relevant` is shared by both local and web-search results — there's no
separate `is_relevant_web` node, since judging relevance is identical either
way; only what happens *after* differs, which the routing functions already
handle via a `web_search_used` flag.

**Fallback ordering, in short:** local `rewrite_question → retrieve` retries
are tried before the web fallback (they're free and reuse the existing
index); Tavily is only called once local retries are exhausted, or
immediately if local retrieval found zero relevant documents to even
rewrite towards. A web-sourced answer that still fails grounding/usefulness
checks can fall back to local retrieval again if local retries aren't
exhausted yet — only the web call itself is capped to a single use per
question, to bound external cost. See `selfrag/graph/routing.py`'s
docstring for the full reasoning.

The always-up-to-date diagram is also rendered live in the app sidebar under
**🗺️ Workflow diagram** (generated straight from the compiled graph via
`app.get_graph().draw_mermaid()`), so it never drifts out of sync with the code.

## Project structure

```
selfrag/                  # Backend package
  config.py                # env/secrets loading (get_env), constants, feature-ready flags
  llm.py                   # chat LLM setup (local Ollama / Ollama Cloud)
  embeddings.py            # Google embeddings + batching/backoff
  vectorstore.py           # global FAISS store: add_documents/clear_documents/get_chunk_count
  chat_store.py            # SQLite persistence for threads/messages/sources
  graph/
    state.py                 # State TypedDict + Pydantic schemas
    nodes.py                  # all node functions
    routing.py                 # all route_after_* conditional-edge functions
    build.py                    # builds + compiles the StateGraph, exposes `app`

ui/                       # Streamlit UI package
  sidebar.py                # config, diagram, source upload, thread list
  chat.py                    # session state, thread switching, history, input handling
  streaming.py                 # app.stream() wrapper + step-name → label mapping
  details.py                    # "🔍 View Self-RAG Details" expander

frontend_self_rag.py     # thin entrypoint wiring the above together

.env.example               # local dev config template
.streamlit/secrets.toml.example  # Streamlit Cloud secrets template
DEPLOYMENT.md               # GitHub + Streamlit Cloud deployment walkthrough
```

## Requirements

- Python 3.10+
- An [Ollama](https://ollama.com) daemon (local) **or** an Ollama Cloud API
  key — this is the chat/generation model.
- A **Google API key** (free tier) for embeddings — retrieval cannot work
  without this. Get one at https://aistudio.google.com/apikey.
- *(Optional)* A **Tavily API key** (free tier) for the web-search fallback.
  Get one at https://app.tavily.com. Without it, the app still works fully
  using only your loaded documents.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and fill in at least GOOGLE_API_KEY
```

If you're running the chat model locally rather than via Ollama Cloud, make
sure Ollama is running and has the model pulled:

```bash
ollama serve
ollama pull qwen2.5:3b
```

## Run

```bash
streamlit run frontend_self_rag.py
```

Then, in the browser tab that opens:

1. Upload a PDF or paste a YouTube URL from the sidebar and click **Add**.
2. Ask a question in the chat box. Watch the live step-by-step status as the
   graph actually executes (retrieval decision, retrieval, relevance
   filtering, grounding check, revision, usefulness check, and — if
   needed — web search).
3. Expand **🔍 View Self-RAG Details** under any answer to see where it came
   from (general knowledge / your documents / the web), the retrieval
   query used, supporting evidence, and the raw retrieved excerpts.
4. Click **🆕 New Chat** to start a fresh thread, or pick any entry under
   **🗂️ Previous chats** in the sidebar to resume it — including its message
   history and the sources that were loaded for it. Threads persist in a
   local SQLite file (`CHAT_DB_PATH`, default `self_rag_chat_history.sqlite3`)
   and survive local app restarts (see `DEPLOYMENT.md` for why this does
   **not** hold on Streamlit Community Cloud).

## Configuration reference

See `.env.example` for the full list. The short version:

| Variable | Required? | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | **Yes** | Embeddings for retrieval (Google `gemini-embedding` models) |
| `OLLAMA_API_KEY` | No | If set, chat model runs on Ollama Cloud instead of local |
| `OLLAMA_CHAT_MODEL` | No | Chat model name (default `qwen2.5:3b`) |
| `TAVILY_API_KEY` | No | Enables the web-search fallback |
| `CHAT_DB_PATH` | No | SQLite file for thread/message/source persistence |
| `LANGCHAIN_*` | No | Optional LangSmith tracing |

Missing `GOOGLE_API_KEY` doesn't crash the app — it shows a clear banner and
disables document upload/retrieval until you add one. Missing
`TAVILY_API_KEY` similarly just disables the web fallback with an info
banner; everything else keeps working.

Every config value is read through `selfrag/config.py`'s `get_env()`, which
checks Streamlit's `st.secrets` first, then `os.environ` — so the exact same
code works locally (via `.env`) and on Streamlit Community Cloud (via its
Secrets dashboard). See `DEPLOYMENT.md` for the full deployment walkthrough.

## Notes on the embedding switch

Switching embeddings changes vector dimensionality, so any previously-built
index is invalid the moment you change embedding models/providers. This app
never persists the FAISS index to disk (it's rebuilt in-memory each time you
load a source), so there's nothing stale to migrate in normal use —
`vectorstore.add_documents()` still defensively clears the index if it ever
detects it was built with a different embedding model than the one
currently configured.

Large PDFs are embedded in batches (`GOOGLE_EMBED_BATCH_SIZE`, default 50
chunks) with exponential backoff on transient rate-limit errors
(`GOOGLE_EMBED_MAX_RETRIES`, default 3 attempts per batch), so hitting the
free tier's rate limit slows ingestion down rather than failing it outright.

## Verifying on your own machine

This project was developed and tested in a sandbox **without** access to a
real Ollama daemon, a real Google API key, Tavily key, or network access to
Ollama/Google/Tavily's endpoints. What *was* verified there:

- Every file compiles (`python -m py_compile`).
- The full package imports cleanly and the graph compiles
  (`from selfrag.graph import app`).
- The Streamlit script runs end-to-end with no exceptions via
  `streamlit.testing.v1.AppTest`, both with placeholder API keys (exercising
  the "configured" code paths) and with no keys at all (exercising the
  graceful-degradation banners).

What was **not** exercised: real LLM/embedding/web-search calls. Before
considering this done, run through the full loop yourself:

1. Start the app, upload a PDF, ask a question you know is answered in it —
   confirm the answer source shown in **🔍 View Self-RAG Details** says
   "Your loaded documents".
2. Ask a question you know is **not** in your documents but is answerable
   from the web, and confirm the web fallback fires (answer source says
   "Web search").
3. Click **🆕 New Chat**, ask something else, then switch back to the first
   thread from the sidebar and confirm its history and sources reappear.
4. Restart the Streamlit process entirely and confirm both threads are
   still listed (proves SQLite persistence, not just session state).
5. Follow `DEPLOYMENT.md` end-to-end on a throwaway GitHub repo + Streamlit
   Cloud app to confirm the secrets flow actually works outside your machine.
