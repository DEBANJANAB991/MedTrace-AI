"""Result model shared by evidence retrieval strategies."""

from dataclasses import dataclass

from medtrace.models import EvidenceRecord


@dataclass(frozen=True)
class RankedEvidence:
    evidence: EvidenceRecord
    score: float
    retrieval_method: str
