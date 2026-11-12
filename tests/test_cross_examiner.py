from dataclasses import replace
from pathlib import Path

import pytest

from medtrace.cross_examiner import CrossExaminer, extract_claims
from medtrace.generation.models import (
    CitedStatement,
    ClaimVerificationStatus,
    GenerationResult,
)
from medtrace.pipeline import run_pipeline
from medtrace.database import SQLiteDatabase
from medtrace.retrieval import KeywordRetriever
from medtrace.retrieval.models import RankedEvidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA = REPOSITORY_ROOT / "data" / "synthetic"


@pytest.fixture
def retriever(tmp_path: Path) -> KeywordRetriever:
    database_path = tmp_path / "cross_examiner.sqlite3"
    run_pipeline(SYNTHETIC_DATA, database_path)
    return KeywordRetriever(SQLiteDatabase(database_path))


def make_response(
    summary: str,
    source_ids: tuple[str, ...] = (),
    *,
    patient_id: str = "SYN-P001",
    key_findings: tuple[CitedStatement, ...] = (),
    uncertainty: str = "The available evidence may be incomplete.",
) -> GenerationResult:
    return GenerationResult(
        patient_id=patient_id,
        question="Synthetic evaluation question",
        summary=summary,
        summary_source_ids=source_ids,
        key_findings=key_findings,
        supporting_evidence=(),
        conflicts=(),
        missing_information=(),
        uncertainty=CitedStatement(uncertainty, ()),
        human_review_required=True,
        abstained=False,
    )


def test_supported_claim_is_matched_to_retrieved_record(
    retriever: KeywordRetriever,
) -> None:
    retrieved = retriever.retrieve("SYN-P001", "blood glucose tracking")
    response = make_response(
        "Blood glucose tracking happens at home. [SYN-N001]",
        ("SYN-N001",),
    )

    result = CrossExaminer().examine(response, retrieved)

    assert result.original_response is response
    assert result.claims[0].verification_status is ClaimVerificationStatus.SUPPORTED
    assert result.claims[0].supporting_source_ids == ("SYN-N001",)
    assert result.claims[0].contradicting_source_ids == ()
    assert result.claims[0].human_review_required is True


def test_no_known_allergies_claim_is_challenged_by_seasonal_allergy_note(
    retriever: KeywordRetriever,
) -> None:
    retrieved = retriever.retrieve("SYN-P001", "known allergies seasonal")
    response = make_response(
        "The patient has no known allergies. [SYN-P001]",
        ("SYN-P001",),
    )

    result = CrossExaminer().examine(response, retrieved)
    claim = result.claims[0]

    assert claim.verification_status is ClaimVerificationStatus.CONTRADICTED
    assert claim.supporting_source_ids == ("SYN-P001",)
    assert claim.contradicting_source_ids == ("SYN-N002",)
    assert claim.cited_source_ids == ("SYN-P001",)
    assert "does not determine which record is correct" in claim.explanation


def test_claim_without_matching_evidence_is_insufficient(
    retriever: KeywordRetriever,
) -> None:
    retrieved = retriever.retrieve("SYN-P003", "fatigue")
    response = make_response(
        "Fatigue has been present for three weeks. [SYN-N004]",
        ("SYN-N004",),
        patient_id="SYN-P003",
    )

    result = CrossExaminer().examine(response, retrieved)

    assert result.claims[0].verification_status is ClaimVerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.claims[0].supporting_source_ids == ()
    assert result.claims[0].contradicting_source_ids == ()


def test_invalid_original_citation_is_preserved_but_never_used_as_evidence(
    retriever: KeywordRetriever,
) -> None:
    retrieved = retriever.retrieve("SYN-P001", "blood glucose tracking")
    response = make_response(
        "Blood glucose tracking happens at home. [SYN-N999]",
        ("SYN-N999",),
    )

    claim = CrossExaminer().examine(response, retrieved).claims[0]

    assert claim.cited_source_ids == ("SYN-N999",)
    assert claim.supporting_source_ids == ("SYN-N001",)
    assert claim.contradicting_source_ids == ()
    assert "SYN-N999" in claim.explanation
    assert all(
        source_id in {item.evidence.record_id for item in retrieved}
        for source_id in (*claim.supporting_source_ids, *claim.contradicting_source_ids)
    )


def test_cross_patient_evidence_is_rejected(
    retriever: KeywordRetriever,
) -> None:
    retrieved = retriever.retrieve("SYN-P001", "blood glucose tracking")
    cross_patient = [
        RankedEvidence(
            evidence=replace(item.evidence, patient_id="SYN-P002"),
            score=item.score,
            retrieval_method=item.retrieval_method,
        )
        for item in retrieved
    ]

    with pytest.raises(ValueError, match="requested patient"):
        CrossExaminer().examine(make_response(""), cross_patient)


def test_uncertainty_and_general_disclaimers_are_not_extracted_as_claims(
    retriever: KeywordRetriever,
) -> None:
    response = replace(
        make_response(
            "This is not medical advice. [SYN-N001]",
            ("SYN-N001",),
            key_findings=(
                CitedStatement(
                    "The available evidence may be incomplete. [SYN-N001]",
                    ("SYN-N001",),
                ),
                CitedStatement(
                    "Blood glucose tracking happens at home. [SYN-N001]",
                    ("SYN-N001",),
                ),
            ),
        ),
        uncertainty=CitedStatement(
            "The records do not establish duration. [SYN-N001]", ("SYN-N001",)
        ),
    )

    claims = extract_claims(response)

    assert len(claims) == 1
    assert claims[0].text == "Blood glucose tracking happens at home."
