import pandas as pd
import pytest

from medtrace.ingestion import HealthcareData
from medtrace.models import IssueType
from medtrace.validation import validate_datasets


def make_data(
    patients: list[dict[str, object]] | None = None,
    medications: list[dict[str, object]] | None = None,
    laboratory_results: list[dict[str, object]] | None = None,
) -> HealthcareData:
    if patients is None:
        patients = [
            {
                "patient_id": "SYN-P001",
                "age": 50,
                "sex": "Female",
                "allergy_information": "Unknown",
                "created_at": "2025-01-01T09:00:00Z",
                "updated_at": "2025-02-01T09:00:00Z",
            }
        ]
    if medications is None:
        medications = [
            {
                "medication_id": "SYN-M001",
                "patient_id": "SYN-P001",
                "medication_name": "Example medication",
                "dosage": "1 unit",
                "status": "active",
                "start_date": "2025-01-01",
                "end_date": None,
                "source": "synthetic-test-feed",
            }
        ]
    if laboratory_results is None:
        laboratory_results = [
            {
                "result_id": "SYN-L001",
                "patient_id": "SYN-P001",
                "test_name": "Example test",
                "value": "1.0",
                "unit": "unit",
                "reference_range": "0-2",
                "observation_date": "2025-01-15",
                "source": "synthetic-test-feed",
            }
        ]

    frames = []
    for records in (patients, medications, laboratory_results):
        frame = pd.DataFrame(records)
        frame["_row_number"] = range(2, len(frame) + 2)
        frames.append(frame)
    return HealthcareData(*frames)


def test_valid_records_have_no_issues() -> None:
    report = validate_datasets(make_data())

    assert report.issues == []
    assert report.has_errors is False


def test_missing_value_is_reported_with_record_and_field() -> None:
    data = make_data(
        patients=[
            {
                "patient_id": "SYN-P001",
                "age": None,
                "sex": "Female",
                "allergy_information": "Unknown",
                "created_at": "2025-01-01",
                "updated_at": "2025-02-01",
            }
        ]
    )

    report = validate_datasets(data)
    issue = next(
        issue for issue in report.issues if issue.issue_type is IssueType.MISSING_VALUE
    )

    assert issue.record_id == "SYN-P001"
    assert issue.field == "age"
    assert issue.row_number == 2
    assert issue.to_dict()["severity"] == "error"


def test_duplicate_record_identifier_is_reported() -> None:
    valid_medication = make_data().medications.iloc[0].drop("_row_number").to_dict()
    duplicate_medication = {**valid_medication, "patient_id": "SYN-P001"}
    data = make_data(medications=[valid_medication, duplicate_medication])

    report = validate_datasets(data)
    duplicate_issues = [
        issue
        for issue in report.issues
        if issue.issue_type is IssueType.DUPLICATE_RECORD
    ]

    assert len(duplicate_issues) == 1
    assert duplicate_issues[0].record_id == "SYN-M001"
    assert duplicate_issues[0].row_number == 3


def test_invalid_date_and_reversed_medication_dates_are_reported() -> None:
    medication = make_data().medications.iloc[0].drop("_row_number").to_dict()
    invalid_date = {**medication, "start_date": "not-a-date"}
    reversed_dates = {
        **medication,
        "medication_id": "SYN-M002",
        "status": "discontinued",
        "start_date": "2025-02-01",
        "end_date": "2025-01-01",
    }
    data = make_data(medications=[invalid_date, reversed_dates])

    report = validate_datasets(data)
    date_issues = [
        issue for issue in report.issues if issue.issue_type is IssueType.INVALID_DATE
    ]

    assert {issue.record_id for issue in date_issues} == {"SYN-M001", "SYN-M002"}


def test_overlapping_medication_status_conflict_is_reported_on_both_records() -> None:
    active = make_data().medications.iloc[0].drop("_row_number").to_dict()
    discontinued = {
        **active,
        "medication_id": "SYN-M002",
        "status": "discontinued",
        "end_date": "2025-02-01",
    }
    data = make_data(medications=[active, discontinued])

    report = validate_datasets(data)
    conflicts = [
        issue
        for issue in report.issues
        if issue.issue_type is IssueType.CONFLICTING_MEDICATION_STATUS
    ]

    assert {issue.record_id for issue in conflicts} == {"SYN-M001", "SYN-M002"}
    assert all("overlapping dates" in issue.explanation for issue in conflicts)


@pytest.mark.parametrize("table", ["medications", "laboratory_results"])
def test_unknown_patient_reference_is_reported(table: str) -> None:
    data = make_data()
    if table == "medications":
        data.medications.loc[0, "patient_id"] = "SYN-P999"
        expected_id = "SYN-M001"
    else:
        data.laboratory_results.loc[0, "patient_id"] = "SYN-P999"
        expected_id = "SYN-L001"

    report = validate_datasets(data)
    unknown_patient = next(
        issue
        for issue in report.issues
        if issue.issue_type is IssueType.UNKNOWN_PATIENT_ID
    )

    assert unknown_patient.record_id == expected_id
    assert "SYN-P999" in unknown_patient.explanation


def test_reference_to_rejected_patient_is_reported() -> None:
    invalid_patient = {
        "patient_id": "SYN-P002",
        "age": None,
        "sex": "Male",
        "allergy_information": "Unknown",
        "created_at": "2025-01-01",
        "updated_at": "2025-02-01",
    }
    medication = make_data().medications.iloc[0].drop("_row_number").to_dict()
    medication["patient_id"] = "SYN-P002"
    data = make_data(
        patients=[
            make_data().patients.iloc[0].drop("_row_number").to_dict(),
            invalid_patient,
        ],
        medications=[medication],
    )

    report = validate_datasets(data)
    reference_issue = next(
        issue
        for issue in report.issues
        if issue.issue_type is IssueType.UNKNOWN_PATIENT_ID
    )

    assert reference_issue.record_id == "SYN-M001"
    assert "SYN-P002" in reference_issue.explanation
