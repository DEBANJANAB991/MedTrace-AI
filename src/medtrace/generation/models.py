"""Structured, provenance-carrying generation response models."""

from dataclasses import asdict, dataclass
from enum import Enum

from medtrace.retrieval.models import RankedEvidence


@dataclass(frozen=True)
class CitedStatement:
    text: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class SupportingEvidence:
    record_id: str
    source: str
    timestamp: str
    content: str


@dataclass(frozen=True)
class GenerationResult:
    patient_id: str
    question: str
    summary: str
    summary_source_ids: tuple[str, ...]
    key_findings: tuple[CitedStatement, ...]
    supporting_evidence: tuple[SupportingEvidence, ...]
    conflicts: tuple[CitedStatement, ...]
    missing_information: tuple[CitedStatement, ...]
    uncertainty: CitedStatement
    human_review_required: bool
    abstained: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of this result."""
        return asdict(self)


class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ClaimAssessment:
    claim_text: str
    cited_source_ids: tuple[str, ...]
    supporting_source_ids: tuple[str, ...]
    contradicting_source_ids: tuple[str, ...]
    verification_status: ClaimVerificationStatus
    explanation: str
    human_review_required: bool = True


@dataclass(frozen=True)
class CrossExaminationResult:
    original_response: GenerationResult
    claims: tuple[ClaimAssessment, ...]
    human_review_required: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return the original answer and independent claim assessments."""
        return {
            "original_response": self.original_response.to_dict(),
            "claims": [asdict(claim) for claim in self.claims],
            "human_review_required": self.human_review_required,
        }


def supporting_evidence_from(
    records: list[RankedEvidence],
) -> tuple[SupportingEvidence, ...]:
    """Build citations from retrieved records, never from model-generated text."""
    return tuple(
        SupportingEvidence(
            record_id=item.evidence.record_id,
            source=item.evidence.source_name,
            timestamp=item.evidence.timestamp,
            content=item.evidence.content,
        )
        for item in records
    )
