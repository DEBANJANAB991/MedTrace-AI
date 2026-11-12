"""Environment-backed configuration for optional LLM providers."""

import os
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None
    base_url: str
    timeout_seconds: float

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        provider = os.getenv("LLM_PROVIDER", "mock").strip().casefold()
        model = os.getenv("LLM_MODEL", "").strip()
        if not model and provider == "mock":
            model = "mock"
        api_key = os.getenv("LLM_API_KEY") or None
        base_url = os.getenv(
            "LLM_BASE_URL", "https://api.openai.com/v1/chat/completions"
        )
        try:
            timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        except ValueError as error:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number") from error

        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number")
        if not model:
            raise ValueError("LLM_MODEL must be set for the selected provider")
        if provider not in {"mock", "openai_compatible"}:
            raise ValueError(
                "LLM_PROVIDER must be 'mock' or 'openai_compatible'"
            )
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
