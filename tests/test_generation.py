import json
from io import BytesIO
from pathlib import Path

import pytest

from medtrace.database import SQLiteDatabase
from medtrace.generation import (
    EvidenceGenerationPipeline,
    LLMSettings,
    MockLLMProvider,
    create_provider,
)
from medtrace.generation.pipeline import GenerationError
from medtrace.generation.providers import (
    OpenAICompatibleProvider,
    ProviderError,
)
from medtrace.pipeline import run_pipeline
from medtrace.retrieval import KeywordRetriever


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA = REPOSITORY_ROOT / "data" / "synthetic"
EVIDENCE_ID = "SYN-N001"


@pytest.fixture
def retriever(tmp_path: Path) -> KeywordRetriever:
    database_path = tmp_path / "generation_test.sqlite3"
    run_pipeline(SYNTHETIC_DATA, database_path)
    return KeywordRetriever(SQLiteDatabase(database_path))


def valid_response(**overrides: object) -> str:
    response: dict[str, object] = {
        "summary": "The note describes home blood-glucose tracking. [SYN-N001]",
        "summary_source_ids": [EVIDENCE_ID],
        "key_findings": [
            {
                "text": "Home blood-glucose tracking is mentioned. [SYN-N001]",
                "source_ids": [EVIDENCE_ID],
            }
        ],
        "conflicts": [],
        "missing_information": [],
        "uncertainty": {
            "text": "The note provides no further details about the tracking. [SYN-N001]",
            "source_ids": [EVIDENCE_ID],
        },
        "abstained": False,
    }
    response.update(overrides)
    return json.dumps(response)


def test_generation_returns_structured_cited_response_and_provenance(
    retriever: KeywordRetriever,
) -> None:
    provider = MockLLMProvider(valid_response())
    pipeline = EvidenceGenerationPipeline(retriever, provider)

    result = pipeline.answer("SYN-P001", "What is mentioned about glucose tracking?")

    assert result.patient_id == "SYN-P001"
    assert result.summary_source_ids == (EVIDENCE_ID,)
    assert result.key_findings[0].source_ids == (EVIDENCE_ID,)
    assert result.supporting_evidence[0].record_id == EVIDENCE_ID
    assert result.supporting_evidence[0].source == "synthetic-clinical-notes-a"
    assert result.supporting_evidence[0].timestamp == "2025-04-15T10:30:00Z"
    assert result.human_review_required is True
    assert result.abstained is False
    prompt_context = json.loads(provider.prompts[0])
    assert prompt_context["patient_id"] == "SYN-P001"
    assert prompt_context["evidence"][0]["record_id"] == EVIDENCE_ID
    assert prompt_context["evidence"][0]["source"] == "synthetic-clinical-notes-a"
    assert prompt_context["evidence"][0]["timestamp"] == "2025-04-15T10:30:00Z"
    assert "blood glucose" in prompt_context["evidence"][0]["content"].casefold()


def test_response_includes_cited_conflicts_and_missing_information(
    retriever: KeywordRetriever,
) -> None:
    response = {
        "summary": "The records contain different allergy wording. [SYN-P001] [SYN-N002]",
        "summary_source_ids": ["SYN-P001", "SYN-N002"],
        "key_findings": [],
        "conflicts": [
            {
                "text": "The registry and note use different allergy wording. [SYN-P001] [SYN-N002]",
                "source_ids": ["SYN-P001", "SYN-N002"],
            }
        ],
        "missing_information": [
            {
                "text": "The note does not add further allergy details. [SYN-N002]",
                "source_ids": ["SYN-N002"],
            }
        ],
        "uncertainty": {
            "text": "The records do not establish whether these descriptions refer to the same event. [SYN-P001] [SYN-N002]",
            "source_ids": ["SYN-P001", "SYN-N002"],
        },
        "abstained": False,
    }
    result = EvidenceGenerationPipeline(
        retriever, MockLLMProvider(json.dumps(response))
    ).answer("SYN-P001", "allergy allergies")

    assert result.conflicts[0].source_ids == ("SYN-P001", "SYN-N002")
    assert result.missing_information[0].source_ids == ("SYN-N002",)
    assert {evidence.record_id for evidence in result.supporting_evidence} >= {
        "SYN-P001",
        "SYN-N002",
    }


def test_no_retrieved_evidence_abstains_without_calling_provider(
    retriever: KeywordRetriever,
) -> None:
    provider = MockLLMProvider(valid_response())
    pipeline = EvidenceGenerationPipeline(retriever, provider)

    result = pipeline.answer("SYN-P002", "blood glucose tracking")

    assert result.abstained is True
    assert result.summary == ""
    assert result.supporting_evidence == ()
    assert result.human_review_required is True
    assert "No relevant records" in result.uncertainty.text
    assert provider.prompts == []


def test_response_rejects_source_ids_outside_retrieved_evidence(
    retriever: KeywordRetriever,
) -> None:
    provider = MockLLMProvider(
        valid_response(summary_source_ids=["SYN-N999"])
    )

    with pytest.raises(GenerationError, match="not present in retrieved evidence"):
        EvidenceGenerationPipeline(retriever, provider).answer(
            "SYN-P001", "glucose tracking"
        )


def test_response_rejects_unknown_inline_citation_even_with_valid_metadata(
    retriever: KeywordRetriever,
) -> None:
    provider = MockLLMProvider(
        valid_response(
            summary="The note mentions glucose tracking. [SYN-N999]",
            summary_source_ids=[EVIDENCE_ID],
        )
    )

    with pytest.raises(GenerationError, match="text cited records"):
        EvidenceGenerationPipeline(retriever, provider).answer(
            "SYN-P001", "glucose tracking"
        )


def test_response_requires_source_ids_for_factual_findings(
    retriever: KeywordRetriever,
) -> None:
    provider = MockLLMProvider(
        valid_response(
            key_findings=[{"text": "The record mentions home tracking.", "source_ids": []}]
        )
    )

    with pytest.raises(GenerationError, match="must cite supporting records"):
        EvidenceGenerationPipeline(retriever, provider).answer(
            "SYN-P001", "glucose tracking"
        )


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "The diagnosis is established.",
        "I recommend that the patient start a medication.",
        "The patient should take more medication.",
        "Consider taking the listed medication.",
        "This is a clinical decision.",
    ],
)
def test_response_rejects_diagnoses_and_clinical_actions(
    retriever: KeywordRetriever,
    unsafe_text: str,
) -> None:
    provider = MockLLMProvider(valid_response(summary=unsafe_text))

    with pytest.raises(GenerationError, match="clinical action instruction"):
        EvidenceGenerationPipeline(retriever, provider).answer(
            "SYN-P001", "glucose tracking"
        )


def test_provider_abstention_is_structured_and_requires_human_review(
    retriever: KeywordRetriever,
) -> None:
    abstaining_response = {
        "summary": "",
        "summary_source_ids": [],
        "key_findings": [],
        "conflicts": [],
        "missing_information": [],
        "uncertainty": {
            "text": "The available note is insufficient to answer the broader question.",
            "source_ids": [],
        },
        "abstained": True,
    }
    result = EvidenceGenerationPipeline(
        retriever, MockLLMProvider(json.dumps(abstaining_response))
    ).answer("SYN-P001", "glucose tracking")

    assert result.abstained is True
    assert result.human_review_required is True
    assert result.supporting_evidence[0].record_id == EVIDENCE_ID


def test_invalid_provider_json_and_provider_errors_are_reported(
    retriever: KeywordRetriever,
) -> None:
    with pytest.raises(GenerationError, match="JSON object"):
        EvidenceGenerationPipeline(
            retriever, MockLLMProvider("not-json")
        ).answer("SYN-P001", "glucose tracking")

    class FailedProvider:
        def generate(self, user_prompt: str) -> str:
            raise ProviderError("provider unavailable")

    with pytest.raises(GenerationError, match="Could not generate"):
        EvidenceGenerationPipeline(retriever, FailedProvider()).answer(
            "SYN-P001", "glucose tracking"
        )


def test_empty_patient_or_question_is_rejected(retriever: KeywordRetriever) -> None:
    pipeline = EvidenceGenerationPipeline(retriever, MockLLMProvider())

    with pytest.raises(ValueError, match="patient_id"):
        pipeline.answer(" ", "question")
    with pytest.raises(ValueError, match="question"):
        pipeline.answer("SYN-P001", " ")


def test_environment_settings_select_mock_or_configure_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = LLMSettings.from_environment()

    assert settings.model == "mock"
    assert isinstance(create_provider(settings), MockLLMProvider)

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        create_provider(LLMSettings.from_environment())

    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1/chat/completions")
    configured = create_provider(LLMSettings.from_environment())
    assert configured.__class__.__name__ == "OpenAICompatibleProvider"


def test_openai_compatible_provider_sends_configured_request_without_live_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_details: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": valid_response()}}]}
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        request_details["url"] = request.full_url
        request_details["authorization"] = request.get_header("Authorization")
        request_details["timeout"] = timeout
        request_details["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("medtrace.generation.providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        api_key="unit-test-secret",
        model="unit-test-model",
        base_url="https://provider.example/v1/chat/completions",
        timeout_seconds=4,
    )

    response = provider.generate('{"question":"synthetic question"}')

    assert json.loads(response)["abstained"] is False
    assert request_details["url"] == "https://provider.example/v1/chat/completions"
    assert request_details["authorization"] == "Bearer unit-test-secret"
    assert request_details["timeout"] == 4
    assert request_details["body"]["model"] == "unit-test-model"
