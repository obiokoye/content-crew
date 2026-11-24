"""Provider configuration and demo-mode detection.

Every provider below has an OpenAI-compatible Chat Completions endpoint, so
CrewAI's built-in OpenAI client works for all of them: we only change the
base URL, key and model. Switch with LLM_PROVIDER in .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# CrewAI sends anonymous telemetry by default; keep this project quiet and offline-friendly.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None
    key_env: str
    default_model: str


PROVIDERS: dict[str, Provider] = {
    # Free tier: https://aistudio.google.com/apikey
    "gemini": Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
    # Free tier: https://console.groq.com/keys
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openai": Provider("openai", None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": Provider("anthropic", "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-haiku-4-5"),
}


@dataclass(frozen=True)
class Settings:
    provider: Provider
    model: str
    api_key: str | None
    demo: bool
    reason: str = ""

    @property
    def label(self) -> str:
        return "demo (scripted)" if self.demo else f"{self.provider.name} · {self.model}"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Read provider settings. Falls back to demo mode when no key is set."""
    env = dict(os.environ) if env is None else env
    name = env.get("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
    if name not in PROVIDERS:
        raise SystemExit(f"Unknown LLM_PROVIDER '{name}'. Choose one of: {', '.join(PROVIDERS)}")
    provider = PROVIDERS[name]
    model = env.get("LLM_MODEL") or provider.default_model
    key = env.get(provider.key_env) or None

    if _truthy(env.get("DEMO_MODE")):
        return Settings(provider, model, key, demo=True, reason="DEMO_MODE=true")
    if not key:
        return Settings(provider, model, None, demo=True, reason=f"no {provider.key_env} set")
    return Settings(provider, model, key, demo=False)


def make_llm(settings: Settings) -> Any:
    """Build the LLM every agent shares: scripted in demo mode, real otherwise."""
    if settings.demo:
        from .demo import ScriptedLLM

        return ScriptedLLM(delay=float(os.getenv("DEMO_DELAY", "0.8")))

    from crewai import LLM

    # The "openai/" prefix selects CrewAI's OpenAI-compatible client.
    return LLM(
        model=f"openai/{settings.model}",
        base_url=settings.provider.base_url,
        api_key=settings.api_key,
        temperature=0.3,
    )
