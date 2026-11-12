import json
from dataclasses import replace
from pathlib import Path

import pytest

from medtrace.database import SQLiteDatabase
from medtrace.demo import build_scenarios, main, run_scenario
from medtrace.evaluation import evaluate_response
from medtrace.pipeline import run_pipeline
from medtrace.retrieval import KeywordRetriever
from medtrace.retrieval.models import RankedEvidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA = REPOSITORY_ROOT / "data" / "synthetic"


@pytest.fixture
def retriever(tmp_path: Path) -> KeywordRetriever:
    database_path = tmp_path / "demo_scenarios.sqlite3"
    run_pipeline(SYNTHETIC_DATA, database_path)
    return KeywordRetriever(SQLiteDatabase(database_path))


@pytest.mark.parametrize(
    "scenario_name",
    [
        "supported_answer",
        "conflicting_records",
        "important_information_missing",
        "no_relevant_records",
        "cross_patient_request",
    ],
)
def test_demo_scenario_passes_deterministic_evaluation(
    retriever: KeywordRetriever,
    scenario_name: str,
) -> None:
    scenario = next(
        scenario for scenario in build_scenarios() if scenario.name == scenario_name
    )

    run = run_scenario(scenario, retriever)

    assert run.evaluation.passed, run.evaluation.to_dict()
    assert run.result.human_review_required is True
    assert all(
        item.evidence.patient_id == scenario.patient_id
        for item in retriever.retrieve(scenario.patient_id, scenario.question)
    )

    if scenario_name == "supported_answer":
        assert not run.result.abstained
        assert run.result.summary_source_ids == ("SYN-N001",)
    elif scenario_name == "conflicting_records":
        assert run.result.conflicts
        assert {source_id for item in run.result.conflicts for source_id in item.source_ids} == {
            "SYN-P001",
            "SYN-N002",
        }
        assert run.cross_examination is not None
        allergy_claim = next(
            claim
            for claim in run.cross_examination.claims
            if "no known allergies" in claim.claim_text.casefold()
        )
        assert allergy_claim.verification_status == "contradicted"
        assert allergy_claim.contradicting_source_ids == ("SYN-N002",)
    elif scenario_name == "important_information_missing":
        assert run.result.abstained
        assert run.result.missing_information
    else:
        assert run.result.abstained
        assert not run.result.supporting_evidence


def test_evaluator_fails_when_citation_is_not_in_retrieved_evidence(
    retriever: KeywordRetriever,
) -> None:
    scenario = next(
        scenario
        for scenario in build_scenarios()
        if scenario.name == "supported_answer"
    )
    run = run_scenario(scenario, retriever)
    retrieved = retriever.retrieve(scenario.patient_id, scenario.question)

    invalid_result = replace(run.result, summary_source_ids=("SYN-N999",))
    evaluation = evaluate_response(
        "tampered_citation",
        scenario.patient_id,
        invalid_result,
        retrieved,
        expect_abstention=False,
    )

    assert not evaluation.passed
    assert not evaluation.checks[0].passed


def test_demo_cli_runs_without_touching_repository_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = REPOSITORY_ROOT / "data" / "medtrace_synthetic.sqlite3"
    database_existed = database_path.exists()
    original_timestamp = (
        database_path.stat().st_mtime_ns if database_existed else None
    )

    result = main(["--data-dir", str(SYNTHETIC_DATA)])
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["all_evaluation_checks_passed"] is True
    assert len(output["scenarios"]) == 5
    if database_existed:
        assert database_path.stat().st_mtime_ns == original_timestamp
    else:
        assert not database_path.exists()


def test_custom_cli_requires_patient_and_question_together() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--patient-id", "SYN-P001"])

    assert error.value.code == 2


def test_evaluator_rejects_retrieved_record_from_another_patient(
    retriever: KeywordRetriever,
) -> None:
    scenario = next(
        scenario
        for scenario in build_scenarios()
        if scenario.name == "supported_answer"
    )
    run = run_scenario(scenario, retriever)
    retrieved = retriever.retrieve(scenario.patient_id, scenario.question)
    leaked = [
        RankedEvidence(
            evidence=replace(item.evidence, patient_id="SYN-P002"),
            score=item.score,
            retrieval_method=item.retrieval_method,
        )
        for item in retrieved
    ]

    evaluation = evaluate_response(
        "tampered_patient_scope",
        scenario.patient_id,
        run.result,
        leaked,
        expect_abstention=False,
    )

    assert not evaluation.checks[1].passed
