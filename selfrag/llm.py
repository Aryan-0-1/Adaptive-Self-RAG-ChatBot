"""
llm.py
------
Chat LLM setup.

Uses ChatOllama for generation. If `OLLAMA_API_KEY` is set, we point it
at Ollama Cloud instead of a local daemon — this is the only way the
chat model works when deployed somewhere (e.g. Streamlit Community
Cloud) that can't reach a local `ollama serve` process.

Cloud and local Ollama use *opposite* model-tag conventions: Ollama
Cloud expects a plain tag (`qwen2.5:3b`), while a local daemon expects
`-cloud`-suffixed tags for cloud-hosted models. We normalize this so
the same `OLLAMA_CHAT_MODEL` env var works either way.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from . import config


def _normalized_model_name(model: str, cloud: bool) -> str:
    if cloud:
        return model[: -len("-cloud")] if model.endswith("-cloud") else model
    return model


def get_chat_llm() -> ChatOllama:
    """Build the chat LLM, routing to Ollama Cloud if a key is configured."""
    cloud = bool(config.OLLAMA_API_KEY)
    model = _normalized_model_name(config.OLLAMA_CHAT_MODEL, cloud)

    if cloud:
        return ChatOllama(
            model=model,
            base_url=config.OLLAMA_HOST,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {config.OLLAMA_API_KEY}"}
            },
        )

    return ChatOllama(model=model, base_url=config.OLLAMA_LOCAL_HOST)


llm = get_chat_llm()
