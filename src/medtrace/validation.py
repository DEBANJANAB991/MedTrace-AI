"""Deterministic data-quality checks for ingested healthcare records."""

from datetime import date
from itertools import combinations
from typing import Any

import pandas as pd

from medtrace.ingestion import HealthcareData
from medtrace.models import IssueType, Severity, ValidationIssue, ValidationReport


TABLE_CONFIG = {
    "patients": {
        "record_id": "patient_id",
        "required_fields": (
            "patient_id",
            "age",
            "sex",
            "allergy_information",
            "created_at",
            "updated_at",
        ),
        "date_fields": ("created_at", "updated_at"),
    },
    "medications": {
        "record_id": "medication_id",
        "required_fields": (
            "medication_id",
            "patient_id",
            "medication_name",
            "dosage",
            "status",
            "start_date",
            "source",
        ),
        "date_fields": ("start_date", "end_date"),
    },
    "laboratory_results": {
        "record_id": "result_id",
        "required_fields": (
            "result_id",
            "patient_id",
            "test_name",
            "value",
            "unit",
            "reference_range",
            "observation_date",
            "source",
        ),
        "date_fields": ("observation_date",),
    },
}


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _parse_date(value: Any) -> date | None:
    if _is_missing(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", format="mixed")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _record_id(row: pd.Series, id_column: str) -> str | None:
    value = row.get(id_column)
    if _is_missing(value):
        return None
    return str(value)


def _row_number(row: pd.Series) -> int:
    if "_row_number" in row:
        return int(row["_row_number"])
    return int(row.name) + 2


def _add_issue(
    report: ValidationReport,
    table: str,
    row: pd.Series,
    issue_type: IssueType,
    explanation: str,
    field: str | None = None,
) -> None:
    id_column = TABLE_CONFIG[table]["record_id"]
    report.issues.append(
        ValidationIssue(
            source_table=table,
            severity=Severity.ERROR,
            issue_type=issue_type,
            record_id=_record_id(row, id_column),
            row_number=_row_number(row),
            explanation=explanation,
            field=field,
        )
    )


def _validate_table_schema(table: str, data_frame: pd.DataFrame) -> None:
    expected_columns = set(TABLE_CONFIG[table]["required_fields"])
    expected_columns.update(TABLE_CONFIG[table]["date_fields"])
    if table == "medications":
        expected_columns.add("end_date")
    missing_columns = sorted(expected_columns.difference(data_frame.columns))
    if missing_columns:
        raise ValueError(
            f"{table} dataset is missing required columns: {', '.join(missing_columns)}"
        )


def _validate_missing_values(
    report: ValidationReport,
    table: str,
    data_frame: pd.DataFrame,
) -> None:
    for _, row in data_frame.iterrows():
        required_fields = TABLE_CONFIG[table]["required_fields"]
        for field in required_fields:
            if field == "end_date":
                continue
            if _is_missing(row[field]):
                _add_issue(
                    report,
                    table,
                    row,
                    IssueType.MISSING_VALUE,
                    f"Required field '{field}' is empty.",
                    field,
                )

        if table == "medications":
            status = row["status"]
            end_date_is_optional = (
                not _is_missing(status) and str(status).strip().casefold() == "active"
            )
            if _is_missing(row["end_date"]) and not end_date_is_optional:
                _add_issue(
                    report,
                    table,
                    row,
                    IssueType.MISSING_VALUE,
                    "Field 'end_date' is required unless medication status is active.",
                    "end_date",
                )


def _validate_dates(
    report: ValidationReport,
    table: str,
    data_frame: pd.DataFrame,
) -> None:
    date_fields = TABLE_CONFIG[table]["date_fields"]
    for _, row in data_frame.iterrows():
        for field in date_fields:
            value = row[field]
            if _is_missing(value):
                continue
            if _parse_date(value) is None:
                _add_issue(
                    report,
                    table,
                    row,
                    IssueType.INVALID_DATE,
                    f"Field '{field}' does not contain a valid date or timestamp.",
                    field,
                )

        if table == "medications":
            start_date = _parse_date(row["start_date"])
            end_date = _parse_date(row["end_date"])
            if start_date is not None and end_date is not None and end_date < start_date:
                _add_issue(
                    report,
                    table,
                    row,
                    IssueType.INVALID_DATE,
                    "Medication end_date is earlier than start_date.",
                    "end_date",
                )


def _validate_duplicate_ids(
    report: ValidationReport,
    table: str,
    data_frame: pd.DataFrame,
) -> None:
    id_column = TABLE_CONFIG[table]["record_id"]
    duplicate_rows = data_frame[id_column].duplicated(keep="first")
    for _, row in data_frame.loc[duplicate_rows].iterrows():
        _add_issue(
            report,
            table,
            row,
            IssueType.DUPLICATE_RECORD,
            f"Record identifier '{_record_id(row, id_column)}' appears more than once.",
            id_column,
        )


def _validate_patient_references(
    report: ValidationReport,
    table: str,
    data_frame: pd.DataFrame,
    patient_ids: set[str],
) -> None:
    for _, row in data_frame.iterrows():
        patient_id = row["patient_id"]
        if _is_missing(patient_id):
            continue
        if str(patient_id) not in patient_ids:
            _add_issue(
                report,
                table,
                row,
                IssueType.UNKNOWN_PATIENT_ID,
                f"Patient identifier '{patient_id}' does not exist in the patients dataset.",
                "patient_id",
            )


def _medication_intervals_overlap(first: pd.Series, second: pd.Series) -> bool:
    first_start = _parse_date(first["start_date"])
    second_start = _parse_date(second["start_date"])
    if first_start is None or second_start is None:
        return False

    first_end = _parse_date(first["end_date"]) or date.max
    second_end = _parse_date(second["end_date"]) or date.max
    return max(first_start, second_start) <= min(first_end, second_end)


def _validate_medication_status_conflicts(
    report: ValidationReport,
    medications: pd.DataFrame,
) -> None:
    usable_medications = medications[
        ~medications["patient_id"].map(_is_missing)
        & ~medications["medication_name"].map(_is_missing)
        & ~medications["status"].map(_is_missing)
    ]
    grouped = usable_medications.groupby(
        [
            usable_medications["patient_id"].astype(str).str.casefold(),
            usable_medications["medication_name"].astype(str).str.strip().str.casefold(),
        ],
        sort=False,
        dropna=False,
    )
    for _, group in grouped:
        for (_, first), (_, second) in combinations(group.iterrows(), 2):
            first_status = str(first["status"]).strip().casefold()
            second_status = str(second["status"]).strip().casefold()
            if (
                first_status != second_status
                and _medication_intervals_overlap(first, second)
            ):
                explanation = (
                    "Medication records for the same patient and medication have "
                    f"overlapping dates but disagree on status "
                    f"('{first['status']}' versus '{second['status']}')."
                )
                _add_issue(
                    report,
                    "medications",
                    first,
                    IssueType.CONFLICTING_MEDICATION_STATUS,
                    explanation,
                    "status",
                )
                _add_issue(
                    report,
                    "medications",
                    second,
                    IssueType.CONFLICTING_MEDICATION_STATUS,
                    explanation,
                    "status",
                )


def validate_datasets(data: HealthcareData) -> ValidationReport:
    """Return record-level issues without modifying the input dataframes."""
    report = ValidationReport()
    data_frames = {
        "patients": data.patients,
        "medications": data.medications,
        "laboratory_results": data.laboratory_results,
    }

    for table, data_frame in data_frames.items():
        _validate_table_schema(table, data_frame)
        _validate_missing_values(report, table, data_frame)
        _validate_dates(report, table, data_frame)
        _validate_duplicate_ids(report, table, data_frame)

    rejected_patient_rows = {
        issue.row_number
        for issue in report.issues
        if issue.source_table == "patients"
    }
    patient_ids = {
        str(row["patient_id"])
        for _, row in data.patients.iterrows()
        if _row_number(row) not in rejected_patient_rows
        and not _is_missing(row["patient_id"])
    }
    _validate_patient_references(
        report, "medications", data.medications, patient_ids
    )
    _validate_patient_references(
        report, "laboratory_results", data.laboratory_results, patient_ids
    )
    _validate_medication_status_conflicts(report, data.medications)
    return report
