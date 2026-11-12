import sqlite3
from pathlib import Path

import pytest

from medtrace.database import SQLiteDatabase
from medtrace.pipeline import run_pipeline


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA = REPOSITORY_ROOT / "data" / "synthetic"


def test_pipeline_persists_only_valid_rows_and_enforces_foreign_keys(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "medtrace_test.sqlite3"

    result = run_pipeline(SYNTHETIC_DATA, database_path)

    assert result.loaded_record_counts == {
        "patients": 3,
        "medications": 1,
        "laboratory_results": 2,
    }
    assert result.validation_report.has_errors
    assert database_path.is_file()

    with sqlite3.connect(database_path) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("patients", "medications", "laboratory_results")
        }
        orphan_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM medications AS medication
            LEFT JOIN patients AS patient USING (patient_id)
            WHERE patient.patient_id IS NULL
            """
        ).fetchone()[0]
        foreign_keys_enabled = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        loaded_source = connection.execute(
            "SELECT source FROM laboratory_results WHERE result_id = ?",
            ("SYN-L001",),
        ).fetchone()[0]

    assert counts == result.loaded_record_counts
    assert orphan_count == 0
    assert foreign_keys_enabled == []
    assert loaded_source == "synthetic-laboratory-feed-a"

    with pytest.raises(sqlite3.IntegrityError):
        with SQLiteDatabase(database_path).connection() as connection:
            connection.execute(
                """
                INSERT INTO medications
                    (medication_id, patient_id, medication_name, dosage, status,
                     start_date, end_date, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "SYN-M-ORPHAN",
                    "SYN-P999",
                    "Example",
                    "1 unit",
                    "active",
                    "2025-01-01",
                    None,
                    "synthetic-test-feed",
                ),
            )
