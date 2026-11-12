"""SQLite schema and connection management for MedTrace synthetic data."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd

from medtrace.ingestion import HealthcareData
from medtrace.models import EvidenceRecord, ValidationReport


SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL,
    allergy_information TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medications (
    medication_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    medication_name TEXT NOT NULL,
    dosage TEXT NOT NULL,
    status TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT,
    source TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS laboratory_results (
    result_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    test_name TEXT NOT NULL,
    value TEXT NOT NULL,
    unit TEXT NOT NULL,
    reference_range TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    note_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    note_text TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
"""


class SQLiteDatabase:
    """Owns SQLite connections and enables foreign-key checks on each one."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    def replace_records(
        self,
        data: HealthcareData,
        report: ValidationReport,
    ) -> dict[str, int]:
        """Replace the database snapshot with rows that passed validation."""
        accepted = {
            "patients": _accepted_rows(data.patients, report, "patients"),
            "medications": _accepted_rows(data.medications, report, "medications"),
            "laboratory_results": _accepted_rows(
                data.laboratory_results, report, "laboratory_results"
            ),
            "clinical_notes": _accepted_rows(
                data.clinical_notes, report, "clinical_notes"
            ),
        }

        with self.connection() as connection:
            connection.executescript(SCHEMA)
            connection.execute("DELETE FROM clinical_notes")
            connection.execute("DELETE FROM laboratory_results")
            connection.execute("DELETE FROM medications")
            connection.execute("DELETE FROM patients")
            _insert_patients(connection, accepted["patients"])
            _insert_medications(connection, accepted["medications"])
            _insert_laboratory_results(
                connection, accepted["laboratory_results"]
            )
            _insert_clinical_notes(connection, accepted["clinical_notes"])

        return {table: len(rows) for table, rows in accepted.items()}

    def fetch_patient_evidence(self, patient_id: str) -> list[EvidenceRecord]:
        """Fetch all searchable evidence for one patient using scoped SQL."""
        with self.connection() as connection:
            patients = connection.execute(
                """
                SELECT patient_id, age, sex, allergy_information,
                       created_at, updated_at
                FROM patients
                WHERE patient_id = ?
                """,
                (patient_id,),
            ).fetchall()
            medications = connection.execute(
                """
                SELECT medication_id, patient_id, medication_name, dosage,
                       status, start_date, end_date, source
                FROM medications
                WHERE patient_id = ?
                """,
                (patient_id,),
            ).fetchall()
            laboratory_results = connection.execute(
                """
                SELECT result_id, patient_id, test_name, value, unit,
                       reference_range, observation_date, source
                FROM laboratory_results
                WHERE patient_id = ?
                """,
                (patient_id,),
            ).fetchall()
            clinical_notes = connection.execute(
                """
                SELECT note_id, patient_id, note_text, observed_at, source
                FROM clinical_notes
                WHERE patient_id = ?
                """,
                (patient_id,),
            ).fetchall()

        evidence: list[EvidenceRecord] = []
        for row in patients:
            evidence.append(
                EvidenceRecord(
                    record_id=row["patient_id"],
                    patient_id=row["patient_id"],
                    record_type="patient",
                    content=(
                        f"Age: {row['age']}; sex: {row['sex']}; "
                        f"allergy information: {row['allergy_information']}"
                    ),
                    source_name="synthetic-patient-registry",
                    timestamp=row["updated_at"],
                    metadata={
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    },
                )
            )
        for row in medications:
            evidence.append(
                EvidenceRecord(
                    record_id=row["medication_id"],
                    patient_id=row["patient_id"],
                    record_type="medication",
                    content=(
                        f"{row['medication_name']}; dosage: {row['dosage']}; "
                        f"status: {row['status']}"
                    ),
                    source_name=row["source"],
                    timestamp=row["start_date"],
                    metadata={
                        "start_date": row["start_date"],
                        "end_date": row["end_date"],
                    },
                )
            )
        for row in laboratory_results:
            evidence.append(
                EvidenceRecord(
                    record_id=row["result_id"],
                    patient_id=row["patient_id"],
                    record_type="laboratory_result",
                    content=(
                        f"{row['test_name']}: {row['value']} {row['unit']}; "
                        f"reference range: {row['reference_range']}"
                    ),
                    source_name=row["source"],
                    timestamp=row["observation_date"],
                    metadata={
                        "observation_date": row["observation_date"],
                        "unit": row["unit"],
                        "reference_range": row["reference_range"],
                    },
                )
            )
        for row in clinical_notes:
            evidence.append(
                EvidenceRecord(
                    record_id=row["note_id"],
                    patient_id=row["patient_id"],
                    record_type="clinical_note",
                    content=row["note_text"],
                    source_name=row["source"],
                    timestamp=row["observed_at"],
                    metadata={"observed_at": row["observed_at"]},
                )
            )
        return evidence


def _accepted_rows(
    data_frame: pd.DataFrame,
    report: ValidationReport,
    table: str,
) -> list[dict[str, object]]:
    rejected_rows = {
        issue.row_number
        for issue in report.issues
        if issue.source_table == table
    }
    accepted = data_frame[
        ~data_frame["_row_number"].isin(rejected_rows)
    ]
    return [
        {
            column: _sqlite_value(value)
            for column, value in row.items()
            if column != "_row_number"
        }
        for row in accepted.to_dict(orient="records")
    ]


def _sqlite_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _insert_patients(
    connection: sqlite3.Connection,
    records: list[dict[str, object]],
) -> None:
    connection.executemany(
        """
        INSERT INTO patients
            (patient_id, age, sex, allergy_information, created_at, updated_at)
        VALUES
            (:patient_id, :age, :sex, :allergy_information, :created_at, :updated_at)
        """,
        records,
    )


def _insert_medications(
    connection: sqlite3.Connection,
    records: list[dict[str, object]],
) -> None:
    connection.executemany(
        """
        INSERT INTO medications
            (medication_id, patient_id, medication_name, dosage, status,
             start_date, end_date, source)
        VALUES
            (:medication_id, :patient_id, :medication_name, :dosage, :status,
             :start_date, :end_date, :source)
        """,
        records,
    )


def _insert_laboratory_results(
    connection: sqlite3.Connection,
    records: list[dict[str, object]],
) -> None:
    connection.executemany(
        """
        INSERT INTO laboratory_results
            (result_id, patient_id, test_name, value, unit, reference_range,
             observation_date, source)
        VALUES
            (:result_id, :patient_id, :test_name, :value, :unit, :reference_range,
             :observation_date, :source)
        """,
        records,
    )


def _insert_clinical_notes(
    connection: sqlite3.Connection,
    records: list[dict[str, object]],
) -> None:
    connection.executemany(
        """
        INSERT INTO clinical_notes
            (note_id, patient_id, note_text, observed_at, source)
        VALUES
            (:note_id, :patient_id, :note_text, :observed_at, :source)
        """,
        records,
    )
