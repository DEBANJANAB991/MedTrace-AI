"""Retrieval-augmented generation orchestration and response validation."""

import json
import logging
import re
from typing import Any, Protocol

from medtrace.generation.models import (
    CitedStatement,
    GenerationResult,
    supporting_evidence_from,
)
from medtrace.generation.prompts import build_user_prompt
from medtrace.generation.providers import LLMProvider, ProviderError
from medtrace.retrieval.models import RankedEvidence


logger = logging.getLogger(__name__)
_CLINICAL_ACTION_PATTERN = re.compile(
    r"\b(?:diagnos\w*|prescrib\w*|recommend\w*|triag\w*)\b|"
    r"\b(?:clinical|medical)\s+decisions?\b|"
    r"\b(?:should|must)\s+(?:take|start|stop|increase|decrease)\b|"
    r"\b(?:suggest|consider|try)\s+(?:that\s+)?(?:the patient\s+)?"
    r"(?:take|takes|taking|start|starting|stop|stopping|increase|increasing|"
    r"decrease|decreasing)\b|"
    r"\b(?:continue|begin)\s+(?:taking|the medication|medication)\b|"
    r"\b(?:take|start|stop|increase|decrease)\s+(?:the\s+)?"
    r"(?:medication|dose|dosage)\b",
    re.IGNORECASE,
)


class EvidenceRetriever(Protocol):
    def retrieve(self, patient_id: str, query: str) -> list[RankedEvidence]:
        """Return ranked evidence scoped to one patient."""
        ...


class GenerationError(RuntimeError):
    """Raised when provider output is invalid or violates evidence constraints."""


def _read_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise GenerationError(f"Provider field '{field}' must be non-empty text")
    return value


def _read_source_ids(
    value: Any,
    field: str,
    valid_source_ids: set[str],
    *,
    required: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(source_id, str) for source_id in value
    ):
        raise GenerationError(f"Provider field '{field}' must be a list of record IDs")
    source_ids = tuple(dict.fromkeys(value))
    if required and not source_ids:
        raise GenerationError(f"Provider field '{field}' must cite supporting records")
    unknown_ids = set(source_ids).difference(valid_source_ids)
    if unknown_ids:
        raise GenerationError(
            f"Provider field '{field}' cited records not present in retrieved evidence"
        )
    return source_ids


def _validate_inline_citations(
    text: str,
    source_ids: tuple[str, ...],
    valid_source_ids: set[str],
) -> None:
    inline_ids = re.findall(r"\[([^\[\]]+)\]", text)
    if any(source_id not in valid_source_ids for source_id in inline_ids):
        raise GenerationError(
            "Provider text cited records not present in retrieved evidence"
        )
    if not set(inline_ids).issubset(source_ids):
        raise GenerationError(
            "Provider text citations must also appear in the structured source IDs"
        )


def _read_statements(
    value: Any,
    field: str,
    valid_source_ids: set[str],
    *,
    required_citations: bool,
) -> tuple[CitedStatement, ...]:
    if not isinstance(value, list):
        raise GenerationError(f"Provider field '{field}' must be a list")
    statements = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise GenerationError(f"Provider field '{field}[{index}]' must be an object")
        text = _read_string(item.get("text"), f"{field}[{index}].text")
        source_ids = _read_source_ids(
            item.get("source_ids"),
            f"{field}[{index}].source_ids",
            valid_source_ids,
            required=required_citations,
        )
        _validate_inline_citations(text, source_ids, valid_source_ids)
        statements.append(CitedStatement(text=text, source_ids=source_ids))
    return tuple(statements)


def _validate_safe_language(texts: list[str]) -> None:
    if any(_CLINICAL_ACTION_PATTERN.search(text) for text in texts):
        raise GenerationError(
            "Provider output contained a diagnosis or clinical action instruction"
        )


class EvidenceGenerationPipeline:
    """Retrieve patient-scoped evidence and produce a cited reviewable response."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        provider: LLMProvider,
    ) -> None:
        self._retriever = retriever
        self._provider = provider

    def answer(self, patient_id: str, question: str) -> GenerationResult:
        if not patient_id.strip():
            raise ValueError("patient_id must not be empty")
        if not question.strip():
            raise ValueError("question must not be empty")

        logger.info("Retrieving evidence for patient-scoped question")
        retrieved = self._retriever.retrieve(patient_id, question)
        if not retrieved:
            logger.info("No relevant evidence retrieved; abstaining without generation")
            return self._abstention(patient_id, question)

        user_prompt = build_user_prompt(patient_id, question, retrieved)
        try:
            raw_response = self._provider.generate(user_prompt)
        except ProviderError as error:
            logger.error("Generation provider failed (%s)", type(error).__name__)
            raise GenerationError("Could not generate an evidence-grounded response") from error

        try:
            payload = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as error:
            logger.error("Generation provider returned invalid JSON")
            raise GenerationError("Provider response must be a JSON object") from error
        return self._parse_response(patient_id, question, retrieved, payload)

    def _parse_response(
        self,
        patient_id: str,
        question: str,
        retrieved: list[RankedEvidence],
        payload: Any,
    ) -> GenerationResult:
        if not isinstance(payload, dict):
            raise GenerationError("Provider response must be a JSON object")

        abstained = payload.get("abstained")
        if not isinstance(abstained, bool):
            raise GenerationError("Provider field 'abstained' must be a boolean")

        valid_source_ids = {item.evidence.record_id for item in retrieved}
        summary = _read_string(payload.get("summary"), "summary", allow_empty=True)
        summary_source_ids = _read_source_ids(
            payload.get("summary_source_ids"),
            "summary_source_ids",
            valid_source_ids,
            required=bool(summary) and not abstained,
        )
        _validate_inline_citations(
            summary, summary_source_ids, valid_source_ids
        )
        if not abstained and not summary:
            raise GenerationError("A non-abstaining response must include a summary")

        key_findings = _read_statements(
            payload.get("key_findings"),
            "key_findings",
            valid_source_ids,
            required_citations=not abstained,
        )
        conflicts = _read_statements(
            payload.get("conflicts"),
            "conflicts",
            valid_source_ids,
            required_citations=not abstained,
        )
        missing_information = _read_statements(
            payload.get("missing_information"),
            "missing_information",
            valid_source_ids,
            required_citations=True,
        )
        uncertainty_payload = payload.get("uncertainty")
        if not isinstance(uncertainty_payload, dict):
            raise GenerationError("Provider field 'uncertainty' must be an object")
        uncertainty = CitedStatement(
            text=_read_string(uncertainty_payload.get("text"), "uncertainty.text"),
            source_ids=_read_source_ids(
                uncertainty_payload.get("source_ids"),
                "uncertainty.source_ids",
                valid_source_ids,
                required=not abstained,
            ),
        )
        _validate_inline_citations(
            uncertainty.text, uncertainty.source_ids, valid_source_ids
        )

        all_statements = [
            summary,
            uncertainty.text,
            *(statement.text for statement in key_findings),
            *(statement.text for statement in conflicts),
            *(statement.text for statement in missing_information),
        ]
        _validate_safe_language(all_statements)

        if abstained and (
            summary or summary_source_ids or key_findings or conflicts
        ):
            raise GenerationError(
                "An abstaining response cannot include summary, findings, or conflicts"
            )
        if abstained:
            logger.info("Provider abstained because evidence was insufficient")

        return GenerationResult(
            patient_id=patient_id,
            question=question,
            summary=summary,
            summary_source_ids=summary_source_ids,
            key_findings=key_findings,
            supporting_evidence=supporting_evidence_from(retrieved),
            conflicts=conflicts,
            missing_information=missing_information,
            uncertainty=uncertainty,
            human_review_required=True,
            abstained=abstained,
        )

    @staticmethod
    def _abstention(patient_id: str, question: str) -> GenerationResult:
        return GenerationResult(
            patient_id=patient_id,
            question=question,
            summary="",
            summary_source_ids=(),
            key_findings=(),
            supporting_evidence=(),
            conflicts=(),
            missing_information=(),
            uncertainty=CitedStatement(
                text="No relevant records were retrieved; the question cannot be answered from the available evidence.",
                source_ids=(),
            ),
            human_review_required=True,
            abstained=True,
        )
