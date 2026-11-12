"""Evidence-grounded summary generation."""

from medtrace.generation.config import LLMSettings
from medtrace.generation.models import (
    CitedStatement,
    GenerationResult,
    SupportingEvidence,
)
from medtrace.generation.pipeline import (
    EvidenceGenerationPipeline,
    EvidenceRetriever,
    GenerationError,
)
from medtrace.generation.providers import (
    LLMProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    create_provider,
)

__all__ = [
    "EvidenceGenerationPipeline",
    "EvidenceRetriever",
    "GenerationError",
    "GenerationResult",
    "CitedStatement",
    "SupportingEvidence",
    "LLMProvider",
    "LLMSettings",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "create_provider",
]
