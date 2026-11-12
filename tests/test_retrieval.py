from pathlib import Path
from typing import Sequence

import pytest

from medtrace.database import SQLiteDatabase
from medtrace.pipeline import run_pipeline
from medtrace.retrieval import KeywordRetriever, SemanticRetriever


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA = REPOSITORY_ROOT / "data" / "synthetic"


@pytest.fixture
def database(tmp_path: Path) -> SQLiteDatabase:
    database_path = tmp_path / "retrieval_test.sqlite3"
    run_pipeline(SYNTHETIC_DATA, database_path)
    return SQLiteDatabase(database_path)


def test_keyword_retrieval_ranks_relevant_notes_and_excludes_irrelevant_records(
    database: SQLiteDatabase,
) -> None:
    results = KeywordRetriever(database).retrieve(
        "SYN-P001", "blood glucose tracking"
    )

    assert results
    assert results[0].evidence.record_id == "SYN-N001"
    assert results[0].evidence.record_type == "clinical_note"
    assert all("glucose" in result.evidence.content.casefold() for result in results)


def test_retrieval_does_not_return_another_patients_records(
    database: SQLiteDatabase,
) -> None:
    results = KeywordRetriever(database).retrieve(
        "SYN-P002", "blood glucose tracking"
    )

    assert results == []


def test_evidence_keeps_record_source_patient_and_timestamps(
    database: SQLiteDatabase,
) -> None:
    results = KeywordRetriever(database).retrieve(
        "SYN-P001", "blood glucose tracking"
    )
    note = results[0].evidence

    assert note.record_id == "SYN-N001"
    assert note.patient_id == "SYN-P001"
    assert note.source_name == "synthetic-clinical-notes-a"
    assert note.timestamp == "2025-04-15T10:30:00Z"
    assert note.metadata["observed_at"] == note.timestamp


def test_structured_evidence_keeps_source_names_and_record_dates(
    database: SQLiteDatabase,
) -> None:
    medication = KeywordRetriever(database).retrieve("SYN-P002", "Amlodipine")[0]
    laboratory_result = KeywordRetriever(database).retrieve(
        "SYN-P001", "HbA1c"
    )[0]
    patient = KeywordRetriever(database).retrieve(
        "SYN-P001", "allergy information"
    )[0]

    assert medication.evidence.record_id == "SYN-M003"
    assert medication.evidence.source_name == "synthetic-medication-feed-a"
    assert medication.evidence.timestamp == "2025-03-10"
    assert laboratory_result.evidence.record_id == "SYN-L001"
    assert laboratory_result.evidence.source_name == "synthetic-laboratory-feed-a"
    assert laboratory_result.evidence.timestamp == "2025-04-12"
    assert patient.evidence.record_type == "patient"
    assert patient.evidence.source_name == "synthetic-patient-registry"
    assert patient.evidence.timestamp == "2025-03-01T10:00:00Z"
    assert patient.evidence.metadata["created_at"] == "2025-01-10T09:00:00Z"


def test_keyword_retrieval_handles_empty_queries_and_no_matches(
    database: SQLiteDatabase,
) -> None:
    retriever = KeywordRetriever(database)

    assert retriever.retrieve("SYN-P001", "  ") == []
    assert retriever.retrieve("SYN-P001", "unmatched-astronomy-term") == []
    assert retriever.retrieve("SYN-P999", "blood glucose") == []


def test_maximum_record_count_is_configurable(database: SQLiteDatabase) -> None:
    results = KeywordRetriever(database, max_records=1).retrieve(
        "SYN-P001", "synthetic note"
    )

    assert len(results) == 1


def test_retriever_rejects_non_positive_maximum(database: SQLiteDatabase) -> None:
    with pytest.raises(ValueError, match="max_records"):
        KeywordRetriever(database, max_records=0)


class FakeEmbedder:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if text == "glucose monitoring":
                vectors.append([1.0, 0.0])
            elif "blood glucose" in text.casefold():
                vectors.append([3.0, 4.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_semantic_retrieval_ranks_by_cosine_similarity(
    database: SQLiteDatabase,
) -> None:
    results = SemanticRetriever(database, FakeEmbedder()).retrieve(
        "SYN-P001", "glucose monitoring"
    )

    assert results[0].evidence.record_id == "SYN-N001"
    assert results[0].retrieval_method == "semantic"
    assert results[0].score == pytest.approx(0.6)


def test_semantic_retrieval_is_patient_scoped_and_empty_without_evidence(
    database: SQLiteDatabase,
) -> None:
    retriever = SemanticRetriever(database, FakeEmbedder())

    assert retriever.retrieve("SYN-P002", "glucose monitoring") == []
    assert retriever.retrieve("SYN-P999", "glucose monitoring") == []
    assert retriever.retrieve("SYN-P001", "  ") == []


def test_semantic_minimum_similarity_filters_unrelated_records(
    database: SQLiteDatabase,
) -> None:
    results = SemanticRetriever(
        database, FakeEmbedder(), min_similarity=0.5
    ).retrieve("SYN-P001", "glucose monitoring")

    assert [result.evidence.record_id for result in results] == ["SYN-N001"]
