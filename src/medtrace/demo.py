"""Run the deterministic end-to-end MedTrace synthetic RAG demonstration."""

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from medtrace.database import SQLiteDatabase
from medtrace.cross_examiner import CrossExaminer
from medtrace.evaluation import EvaluationResult, evaluate_response
from medtrace.generation import EvidenceGenerationPipeline, MockLLMProvider
from medtrace.generation.models import CrossExaminationResult, GenerationResult
from medtrace.pipeline import run_pipeline
from medtrace.retrieval import KeywordRetriever
from medtrace.retrieval.models import RankedEvidence


@dataclass(frozen=True)
class DemoScenario:
    name: str
    patient_id: str
    question: str
    provider_response: str | None
    expect_abstention: bool
    require_conflict: bool = False
    require_missing_information: bool = False
    require_no_relevant_evidence: bool = False


@dataclass(frozen=True)
class ScenarioRun:
    scenario: str
    patient_id: str
    question: str
    result: GenerationResult
    evaluation: EvaluationResult
    cross_examination: CrossExaminationResult | None = None

    def to_dict(self) -> dict[str, object]:
        output = {
            "scenario": self.scenario,
            "patient_id": self.patient_id,
            "question": self.question,
            "original_generated_answer": self.result.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }
        if self.cross_examination is not None:
            output["cross_examination"] = self.cross_examination.to_dict()
        return output


def _response(
    summary: str,
    summary_source_ids: list[str],
    key_findings: list[dict[str, object]],
    conflicts: list[dict[str, object]],
    missing_information: list[dict[str, object]],
    uncertainty: str,
    uncertainty_source_ids: list[str],
    *,
    abstained: bool = False,
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "summary_source_ids": summary_source_ids,
            "key_findings": key_findings,
            "conflicts": conflicts,
            "missing_information": missing_information,
            "uncertainty": {
                "text": uncertainty,
                "source_ids": uncertainty_source_ids,
            },
            "abstained": abstained,
        }
    )


def build_scenarios() -> tuple[DemoScenario, ...]:
    """Return fixed synthetic evaluation cases and mock provider responses."""
    supported_note = "SYN-N001"
    conflict_registry = "SYN-P001"
    conflict_note = "SYN-N002"
    fatigue_note = "SYN-N004"

    return (
        DemoScenario(
            name="supported_answer",
            patient_id="SYN-P001",
            question="Was blood glucose tracking mentioned?",
            provider_response=_response(
                "The note says blood glucose is tracked at home and routine monitoring was discussed. [SYN-N001]",
                [supported_note],
                [
                    {
                        "text": "The note mentions home blood-glucose tracking. [SYN-N001]",
                        "source_ids": [supported_note],
                    }
                ],
                [],
                [],
                "The note does not provide additional tracking details. [SYN-N001]",
                [supported_note],
            ),
            expect_abstention=False,
        ),
        DemoScenario(
            name="conflicting_records",
            patient_id="SYN-P001",
            question="What do the records say about known allergies?",
            provider_response=_response(
                "The patient has no known allergies. [SYN-P001]",
                [conflict_registry],
                [
                    {
                        "text": "The patient has no known allergies. [SYN-P001]",
                        "source_ids": [conflict_registry],
                    },
                ],
                [
                    {
                        "text": "The registry and note differ on allergy information. [SYN-P001] [SYN-N002]",
                        "source_ids": [conflict_registry, conflict_note],
                    }
                ],
                [],
                "The records do not establish which description is current. [SYN-P001] [SYN-N002]",
                [conflict_registry, conflict_note],
            ),
            expect_abstention=False,
            require_conflict=True,
        ),
        DemoScenario(
            name="important_information_missing",
            patient_id="SYN-P003",
            question="How long has the fatigue been present?",
            provider_response=_response(
                "",
                [],
                [],
                [],
                [
                    {
                        "text": "The note mentions fatigue but does not state its duration. [SYN-N004]",
                        "source_ids": [fatigue_note],
                    }
                ],
                "The available note is insufficient to determine duration. [SYN-N004]",
                [fatigue_note],
                abstained=True,
            ),
            expect_abstention=True,
            require_missing_information=True,
        ),
        DemoScenario(
            name="no_relevant_records",
            patient_id="SYN-P002",
            question="What do the records say about blood glucose tracking?",
            provider_response=None,
            expect_abstention=True,
            require_no_relevant_evidence=True,
        ),
        DemoScenario(
            name="cross_patient_request",
            patient_id="SYN-P002",
            question="Can you show SYN-P001 glucose?",
            provider_response=None,
            expect_abstention=True,
            require_no_relevant_evidence=True,
        ),
    )


def run_scenario(
    scenario: DemoScenario,
    retriever: KeywordRetriever,
) -> ScenarioRun:
    retrieved = retriever.retrieve(scenario.patient_id, scenario.question)
    provider = MockLLMProvider(scenario.provider_response)
    result = EvidenceGenerationPipeline(retriever, provider).answer(
        scenario.patient_id, scenario.question
    )
    evaluation = evaluate_response(
        scenario.name,
        scenario.patient_id,
        result,
        retrieved,
        expect_abstention=scenario.expect_abstention,
        require_conflict=scenario.require_conflict,
        require_missing_information=scenario.require_missing_information,
        require_no_relevant_evidence=scenario.require_no_relevant_evidence,
    )
    cross_examination = None
    if scenario.name == "conflicting_records":
        cross_examination = CrossExaminer().examine(result, retrieved)
    return ScenarioRun(
        scenario=scenario.name,
        patient_id=scenario.patient_id,
        question=scenario.question,
        result=result,
        evaluation=evaluation,
        cross_examination=cross_examination,
    )


def _render_single_answer(result: GenerationResult) -> dict[str, object]:
    return {
        "patient_id": result.patient_id,
        "question": result.question,
        "summary": result.summary,
        "summary_source_ids": list(result.summary_source_ids),
        "key_findings": [
            {"text": finding.text, "source_ids": list(finding.source_ids)}
            for finding in result.key_findings
        ],
        "supporting_evidence": [
            {
                "record_id": evidence.record_id,
                "source": evidence.source,
                "timestamp": evidence.timestamp,
                "content": evidence.content,
            }
            for evidence in result.supporting_evidence
        ],
        "conflicts": [
            {"text": item.text, "source_ids": list(item.source_ids)}
            for item in result.conflicts
        ],
        "missing_information": [
            {"text": item.text, "source_ids": list(item.source_ids)}
            for item in result.missing_information
        ],
        "uncertainty": {
            "text": result.uncertainty.text,
            "source_ids": list(result.uncertainty.source_ids),
        },
        "human_review_required": result.human_review_required,
        "abstained": result.abstained,
    }


def main(argv: Sequence[str] | None = None) -> int:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run MedTrace's synthetic, mock-provider end-to-end demo."
    )
    parser.add_argument("--data-dir", type=Path, default=repository_root / "data" / "synthetic")
    parser.add_argument(
        "--patient-id",
        help="Run one question instead of the full scenario suite; requires --question.",
    )
    parser.add_argument("--question", help="Natural-language question for --patient-id.")
    arguments = parser.parse_args(argv)
    if bool(arguments.patient_id) != bool(arguments.question):
        parser.error("--patient-id and --question must be supplied together")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    with tempfile.TemporaryDirectory(prefix="medtrace-demo-") as directory:
        database_path = Path(directory) / "medtrace.sqlite3"
        pipeline_result = run_pipeline(arguments.data_dir, database_path)
        retriever = KeywordRetriever(SQLiteDatabase(database_path), max_records=5)

        if arguments.patient_id:
            question = arguments.question
            retrieved = retriever.retrieve(arguments.patient_id, question)
            result = EvidenceGenerationPipeline(
                retriever, MockLLMProvider()
            ).answer(arguments.patient_id, question)
            evaluation = evaluate_response(
                "custom_question",
                arguments.patient_id,
                result,
                retrieved,
                expect_abstention=True,
            )
            output: dict[str, object] = {
                "loaded_record_counts": pipeline_result.loaded_record_counts,
                "validation_issue_count": len(pipeline_result.validation_report.issues),
                "mode": "abstaining mock provider; no LLM call",
                "answer": _render_single_answer(result),
                "evaluation": evaluation.to_dict(),
            }
        else:
            runs = [
                run_scenario(scenario, retriever)
                for scenario in build_scenarios()
            ]
            output = {
                "loaded_record_counts": pipeline_result.loaded_record_counts,
                "validation_issue_count": len(pipeline_result.validation_report.issues),
                "mode": "fixed synthetic scenarios with mock provider responses",
                "scenarios": [run.to_dict() for run in runs],
                "all_evaluation_checks_passed": all(
                    run.evaluation.passed for run in runs
                ),
            }

    print(json.dumps(output, indent=2))
    return 0 if output.get("all_evaluation_checks_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
