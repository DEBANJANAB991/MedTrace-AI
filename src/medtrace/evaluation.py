"""Deterministic structural checks for synthetic RAG demonstration runs."""

from dataclasses import asdict, dataclass

from medtrace.generation.models import GenerationResult
from medtrace.retrieval.models import RankedEvidence


@dataclass(frozen=True)
class EvaluationCheck:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class EvaluationResult:
    scenario: str
    checks: tuple[EvaluationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _citation_ids(result: GenerationResult) -> set[str]:
    citations = set(result.summary_source_ids)
    statements = (
        *result.key_findings,
        *result.conflicts,
        *result.missing_information,
        result.uncertainty,
    )
    for statement in statements:
        citations.update(statement.source_ids)
    return citations


def evaluate_response(
    scenario: str,
    requested_patient_id: str,
    result: GenerationResult,
    retrieved: list[RankedEvidence],
    *,
    expect_abstention: bool,
    require_conflict: bool = False,
    require_missing_information: bool = False,
    require_no_relevant_evidence: bool = False,
) -> EvaluationResult:
    """Check provenance, patient scope, and expected demo-level behavior.

    These checks validate response structure and scenario expectations only;
    they do not establish clinical safety or factual correctness.
    """
    evidence_by_id = {item.evidence.record_id: item.evidence for item in retrieved}
    citations = _citation_ids(result)
    supporting_ids = {item.record_id for item in result.supporting_evidence}

    checks = (
        EvaluationCheck(
            name="citations_exist_in_retrieved_evidence",
            passed=citations.issubset(evidence_by_id)
            and supporting_ids.issubset(evidence_by_id),
            details="Every cited and displayed evidence ID must be retrieved.",
        ),
        EvaluationCheck(
            name="citations_belong_to_requested_patient",
            passed=result.patient_id == requested_patient_id
            and all(
                evidence.patient_id == requested_patient_id
                for evidence in evidence_by_id.values()
            )
            and all(
                source_id in evidence_by_id
                and evidence_by_id[source_id].patient_id == requested_patient_id
                for source_id in citations
            ),
            details="Retrieved and cited records must belong to the requested patient.",
        ),
        EvaluationCheck(
            name="no_relevant_evidence_abstains",
            passed=bool(retrieved) or result.abstained,
            details="A response with no retrieved evidence must abstain.",
        ),
        EvaluationCheck(
            name="abstention_matches_scenario",
            passed=result.abstained is expect_abstention
            and (
                not expect_abstention
                or (
                    not result.summary
                    and not result.key_findings
                    and not result.conflicts
                )
            ),
            details="The response must abstain for expected insufficient-evidence cases.",
        ),
        EvaluationCheck(
            name="expected_retrieval_coverage",
            passed=not require_no_relevant_evidence or not retrieved,
            details="Scenario requires that no relevant evidence is retrieved.",
        ),
        EvaluationCheck(
            name="conflict_identified",
            passed=not require_conflict or bool(result.conflicts),
            details="Conflicting scenario must include at least one conflict statement.",
        ),
        EvaluationCheck(
            name="missing_information_identified",
            passed=not require_missing_information
            or bool(result.missing_information),
            details="Missing-information scenario must describe the evidence gap.",
        ),
        EvaluationCheck(
            name="human_review_required",
            passed=result.human_review_required is True,
            details="Every generated or abstaining response must require human review.",
        ),
    )
    return EvaluationResult(scenario=scenario, checks=checks)
