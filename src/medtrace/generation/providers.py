"""LLM provider abstraction, safe mock, and opt-in OpenAI-compatible adapter."""

import json
import logging
from math import isfinite
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from medtrace.generation.config import LLMSettings
from medtrace.generation.prompts import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when an LLM provider cannot return a usable response."""


class LLMProvider(Protocol):
    def generate(self, user_prompt: str) -> str:
        """Generate a response from the supplied evidence prompt.

        Provider implementations should raise ``ProviderError`` for failures.
        """
        ...


class MockLLMProvider:
    """Deterministic provider for local development and tests."""

    def __init__(self, response: str | None = None) -> None:
        self.response = response or json.dumps(
            {
                "summary": "",
                "summary_source_ids": [],
                "key_findings": [],
                "conflicts": [],
                "missing_information": [],
                "uncertainty": {
                    "text": "The mock provider did not generate a clinical summary.",
                    "source_ids": [],
                },
                "abstained": True,
            }
        )
        self.prompts: list[str] = []

    def generate(self, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return self.response


class OpenAICompatibleProvider:
    """Small standard-library adapter for chat-completions-compatible APIs."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 30,
    ) -> None:
        if not api_key.strip():
            raise ValueError("An API key is required for this LLM provider")
        if not model.strip():
            raise ValueError("A model name is required for this LLM provider")
        if not base_url.startswith("https://"):
            raise ValueError("LLM_BASE_URL must use HTTPS")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    def generate(self, user_prompt: str) -> str:
        request_body = json.dumps(
            {
                "model": self._model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        request = Request(
            self._base_url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                provider_payload = json.loads(response.read().decode("utf-8"))
            content = provider_payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("Provider content is not text")
            return content
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
            logger.error("LLM provider request failed (%s)", type(error).__name__)
            raise ProviderError("LLM provider request failed") from error


def create_provider(settings: LLMSettings) -> LLMProvider:
    """Create the configured provider without making a network request."""
    if settings.provider == "mock":
        return MockLLMProvider()
    if settings.provider == "openai_compatible":
        if settings.api_key is None:
            raise ValueError(
                "LLM_API_KEY is required when LLM_PROVIDER=openai_compatible"
            )
        return OpenAICompatibleProvider(
            api_key=settings.api_key,
            model=settings.model,
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.provider}")
