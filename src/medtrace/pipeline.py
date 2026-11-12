"""Orchestration for CSV ingestion, validation, and SQLite persistence."""

from pathlib import Path

from medtrace.database import SQLiteDatabase
from medtrace.ingestion import load_csv_datasets
from medtrace.models import PipelineResult
from medtrace.validation import validate_datasets


def run_pipeline(data_directory: Path, database_path: Path) -> PipelineResult:
    """Load synthetic CSVs, report issues, and persist accepted records."""
    data = load_csv_datasets(data_directory)
    validation_report = validate_datasets(data)
    loaded_record_counts = SQLiteDatabase(database_path).replace_records(
        data, validation_report
    )
    return PipelineResult(
        validation_report=validation_report,
        loaded_record_counts=loaded_record_counts,
    )
