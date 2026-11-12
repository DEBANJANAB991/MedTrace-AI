"""Typed result models shared by ingestion and validation."""

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueType(str, Enum):
    MISSING_VALUE = "missing_value"
    DUPLICATE_RECORD = "duplicate_record"
    INVALID_DATE = "invalid_date"
    CONFLICTING_MEDICATION_STATUS = "conflicting_medication_status"
    UNKNOWN_PATIENT_ID = "unknown_patient_id"


@dataclass(frozen=True)
class ValidationIssue:
    source_table: str
    severity: Severity
    issue_type: IssueType
    record_id: str | None
    row_number: int
    explanation: str
    field: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-friendly representation of this issue."""
        return {
            "source_table": self.source_table,
            "severity": self.severity.value,
            "issue_type": self.issue_type.value,
            "record_id": self.record_id,
            "row_number": self.row_number,
            "field": self.field,
            "explanation": self.explanation,
        }


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    def to_list(self) -> list[dict[str, str | int | None]]:
        return [issue.to_dict() for issue in self.issues]


@dataclass(frozen=True)
class PipelineResult:
    validation_report: ValidationReport
    loaded_record_counts: dict[str, int]
