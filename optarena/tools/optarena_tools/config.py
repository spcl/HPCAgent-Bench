"""Environment-backed settings for the tools service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("TOOLS_HOST", "0.0.0.0")
    port: int = _int_env("TOOLS_PORT", 8801)
    log_level: str = os.getenv("TOOLS_LOG_LEVEL", "info")

    serpapi_api_key: str | None = os.getenv("SERPAPI_API_KEY") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    web_research_agent_path: str | None = os.getenv("WEB_RESEARCH_AGENT_PATH") or None
    web_research_agent_timeout_seconds: int = _int_env("WEB_RESEARCH_AGENT_TIMEOUT_SECONDS", 120)
    agentic_internet_timeout_seconds: int = _int_env("AGENTIC_INTERNET_TIMEOUT_SECONDS", 300)

    judge_url: str = os.getenv("OPTARENA_JUDGE_URL", "http://judge:8800").rstrip("/")
    default_language: str = os.getenv("OPTARENA_DEFAULT_LANGUAGE", "c")
    default_preset: str = os.getenv("OPTARENA_DEFAULT_PRESET", "S")


settings = Settings()
