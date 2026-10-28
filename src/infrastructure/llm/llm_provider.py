import functools
import os
import urllib.request
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from loguru import logger

from infrastructure.config import (
    ROUTER_MODEL,
    ROUTER_PROVIDER,
    EXTRACTOR_MODEL,
    EXTRACTOR_PROVIDER,
    GROQ_BASE_URL,
    CHAT_MODEL,
    CHAT_PROVIDER,
    FAST_CHAT_MODEL,
    FAST_CHAT_PROVIDER,
    OPENROUTER_BASE_URL,
    get_api_key,
)

_GROQ_TO_OPENROUTER = {
    "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-8b-instant": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
}
_DEFAULT_OPENROUTER_FALLBACK = os.getenv(
    "OPENROUTER_FALLBACK_MODEL", "meta-llama/llama-3.3-70b-instruct"
)

def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
def _groq_reachable() -> bool:
    """Probe Groq once per process (result cached).

    Returns False if Groq's edge blocks this host (403/connection error), so
    callers can fall back to OpenRouter. A browser User-Agent is sent so the
    probe mirrors the real httpx client (the block is IP-based, not UA-based).
    """
    if _env_true("LLM_DISABLE_GROQ_FALLBACK"):
        return True   # operator forced Groq; never fall back
    if _env_true("LLM_FORCE_OPENROUTER"):
        logger.warning("LLM_FORCE_OPENROUTER set — routing Groq roles to OpenRouter")
        return False
    try:
        req = urllib.request.Request(
            GROQ_BASE_URL.rstrip("/") + "/models",
            headers={
                "Authorization": f"Bearer {get_api_key('groq')}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        urllib.request.urlopen(req, timeout=4)
        return True
    except Exception as exc:  # noqa: BLE001 — any failure means "unreachable"
        logger.warning(
            "Groq unreachable ({}). Falling back to OpenRouter for Groq-backed "
            "LLMs (router / extractor / fast chat / voice).",
            getattr(exc, "code", exc),
        )
        return False
    



def _build_llm(
    model: str,
    provider: str,
    temperature: float = 0,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatOpenAI:
  
    if provider == "groq" and not _groq_reachable():
        fallback_model = _GROQ_TO_OPENROUTER.get(model, _DEFAULT_OPENROUTER_FALLBACK)
        logger.info("Groq→OpenRouter failover: {} → {}", model, fallback_model)
        provider, model = "openrouter", fallback_model

    llm_kwargs: dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        streaming=streaming,
        max_tokens=max_tokens,
        **kwargs,
    )

    if provider == "openrouter":
        llm_kwargs["openai_api_base"] = OPENROUTER_BASE_URL
        llm_kwargs["openai_api_key"] = get_api_key("openrouter")
    elif provider == "groq":
        llm_kwargs["openai_api_base"] = GROQ_BASE_URL
        llm_kwargs["openai_api_key"] = get_api_key("groq")
    elif provider == "openai":
        llm_kwargs["openai_api_key"] = get_api_key("openai")

    return ChatOpenAI(**llm_kwargs)

