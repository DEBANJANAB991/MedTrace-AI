"""Deterministic claim-level comparison against retrieved evidence."""

import re
from dataclasses import dataclass

from medtrace.generation.models import (
    ClaimAssessment,
    ClaimVerificationStatus,
    CrossExaminationResult,
    GenerationResult,
)
from medtrace.retrieval.models import RankedEvidence


_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_CLAUSE_PATTERN = re.compile(r"\b(?:while|but|however)\b", re.IGNORECASE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_DISCLAIMER_OR_UNCERTAINTY = re.compile(
    r"\b(?:not medical advice|informational purposes|human review required|"
    r"cannot determine|cannot conclude|insufficient evidence|uncertain|"
    r"may be|might be|could be|appears to be|does not establish)\b",
    re.IGNORECASE,
)
_ALLERGY_TERM = re.compile(r"\b(?:allerg(?:y|ies|ic|ics))\b", re.IGNORECASE)
_NEGATIVE_ALLERGY = re.compile(
    r"\b(?:no|without|negative for|denies)\b.{0,30}\b(?:allerg|allergic)\w*\b|"
    r"\bno known allergies\b",
    re.IGNORECASE,
)
_POSITIVE_ALLERGY = re.compile(
    r"\b(?:seasonal|food|drug|medication|penicillin)\s+allerg(?:y|ies)\b|"
    r"\ballergic to\b|\b(?:has|with|reports?)\s+allerg(?:y|ies)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "based", "by", "describes",
    "does", "for", "from", "has", "have", "in", "information", "is", "it",
    "lists", "mentioned", "note", "of", "on", "patient", "record", "records",
    "reported", "reports", "says", "shows", "states", "that", "the", "this",
    "to", "was", "were", "with", "happens", "been", "present",
}


@dataclass(frozen=True)
class _ExtractedClaim:
    text: str
    cited_source_ids: tuple[str, ...]


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOP_WORDS
    }


def _extract_from_statement(
    text: str,
    source_ids: tuple[str, ...],
) -> list[_ExtractedClaim]:
    claims = []
    for sentence in _SENTENCE_PATTERN.split(text.strip()):
        if _DISCLAIMER_OR_UNCERTAINTY.search(sentence):
            continue
        clauses = _CLAUSE_PATTERN.split(sentence)
        for clause in clauses:
            clean_text = _CITATION_PATTERN.sub("", clause).strip(" ,;:-")
            if not clean_text or not _content_tokens(clean_text):
                continue
            claims.append(_ExtractedClaim(clean_text, source_ids))
    return claims


def extract_claims(response: GenerationResult) -> tuple[_ExtractedClaim, ...]:
    """Extract factual-looking clauses; omit the dedicated uncertainty field."""
    extracted = []
    if response.summary:
        extracted.extend(
            _extract_from_statement(response.summary, response.summary_source_ids)
        )
    statement_groups = (
        response.key_findings,
        response.conflicts,
        response.missing_information,
    )
    for statements in statement_groups:
        for statement in statements:
            extracted.extend(
                _extract_from_statement(statement.text, statement.source_ids)
            )

    unique_claims: dict[tuple[str, tuple[str, ...]], _ExtractedClaim] = {}
    for claim in extracted:
        key = (claim.text.casefold(), claim.cited_source_ids)
        unique_claims.setdefault(key, claim)
    return tuple(unique_claims.values())


def _allergy_polarity(text: str) -> bool | None:
    if not _ALLERGY_TERM.search(text):
        return None
    if _NEGATIVE_ALLERGY.search(text):
        return False
    if _POSITIVE_ALLERGY.search(text):
        return True
    return None


def _supports_claim(claim_text: str, evidence_text: str) -> bool:
    claim_polarity = _allergy_polarity(claim_text)
    evidence_polarity = _allergy_polarity(evidence_text)
    if claim_polarity is not None:
        return evidence_polarity is claim_polarity

    claim_tokens = _content_tokens(claim_text)
    return bool(claim_tokens) and claim_tokens.issubset(_content_tokens(evidence_text))


def _contradicts_claim(claim_text: str, evidence_text: str) -> bool:
    claim_polarity = _allergy_polarity(claim_text)
    evidence_polarity = _allergy_polarity(evidence_text)
    return (
        claim_polarity is not None
        and evidence_polarity is not None
        and claim_polarity != evidence_polarity
    )


def _assess_claim(
    claim: _ExtractedClaim,
    retrieved: list[RankedEvidence],
) -> ClaimAssessment:
    supporting_ids = []
    contradicting_ids = []
    for item in retrieved:
        evidence = item.evidence
        if _supports_claim(claim.text, evidence.content):
            supporting_ids.append(evidence.record_id)
        if _contradicts_claim(claim.text, evidence.content):
            contradicting_ids.append(evidence.record_id)

    invalid_citations = set(claim.cited_source_ids).difference(
        item.evidence.record_id for item in retrieved
    )
    if contradicting_ids:
        status = ClaimVerificationStatus.CONTRADICTED
        explanation = (
            "Retrieved records contain content with an explicitly opposite "
            "allergy polarity; the verifier does not determine which record is correct."
        )
    elif supporting_ids:
        status = ClaimVerificationStatus.SUPPORTED
        explanation = (
            "The listed retrieved records contain matching lexical evidence; "
            "this deterministic match is not a clinical correctness judgment."
        )
    else:
        status = ClaimVerificationStatus.INSUFFICIENT_EVIDENCE
        explanation = "No retrieved record sufficiently matches this claim."

    if invalid_citations:
        invalid_ids = ", ".join(sorted(invalid_citations))
        explanation += (
            f" Original response citation(s) not present in retrieved evidence "
            f"were not used: {invalid_ids}."
        )

    return ClaimAssessment(
        claim_text=claim.text,
        cited_source_ids=claim.cited_source_ids,
        supporting_source_ids=tuple(supporting_ids),
        contradicting_source_ids=tuple(contradicting_ids),
        verification_status=status,
        explanation=explanation,
    )


class CrossExaminer:
    """Independently compare generated claims with patient-scoped evidence."""

    def examine(
        self,
        response: GenerationResult,
        retrieved: list[RankedEvidence],
    ) -> CrossExaminationResult:
        for item in retrieved:
            if item.evidence.patient_id != response.patient_id:
                raise ValueError(
                    "Cross-examination evidence must belong to the requested patient"
                )

        claims = tuple(
            _assess_claim(claim, retrieved) for claim in extract_claims(response)
        )
        return CrossExaminationResult(
            original_response=response,
            claims=claims,
            human_review_required=True,
        )
