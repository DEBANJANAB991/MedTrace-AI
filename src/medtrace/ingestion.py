"""CSV ingestion for the synthetic healthcare demonstration datasets."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HealthcareData:
    patients: pd.DataFrame
    medications: pd.DataFrame
    laboratory_results: pd.DataFrame
    clinical_notes: pd.DataFrame


DATASET_FILES = {
    "patients": "patients.csv",
    "medications": "medications.csv",
    "laboratory_results": "laboratory_results.csv",
    "clinical_notes": "clinical_notes.csv",
}

IDENTIFIER_COLUMNS = {
    "patients": {"patient_id": "string"},
    "medications": {"medication_id": "string", "patient_id": "string"},
    "laboratory_results": {"result_id": "string", "patient_id": "string"},
    "clinical_notes": {"note_id": "string", "patient_id": "string"},
}


def load_csv_datasets(data_directory: Path) -> HealthcareData:
    """Load the three CSV files and retain their original CSV row numbers."""
    data_frames: dict[str, pd.DataFrame] = {}
    for dataset_name, file_name in DATASET_FILES.items():
        file_path = data_directory / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"Required {dataset_name} CSV not found: {file_path}")

        data_frame = pd.read_csv(
            file_path,
            dtype=IDENTIFIER_COLUMNS[dataset_name],
            keep_default_na=True,
        )
        data_frame["_row_number"] = range(2, len(data_frame) + 2)
        data_frames[dataset_name] = data_frame

    return HealthcareData(
        patients=data_frames["patients"],
        medications=data_frames["medications"],
        laboratory_results=data_frames["laboratory_results"],
        clinical_notes=data_frames["clinical_notes"],
    )
