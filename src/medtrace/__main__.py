"""Command-line entry point for the synthetic ingestion demonstration."""

import argparse
import json
from pathlib import Path

from medtrace.pipeline import run_pipeline


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Validate synthetic healthcare CSVs and load clean records into SQLite."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repository_root / "data" / "synthetic",
        help="Directory containing patients.csv, medications.csv, and laboratory_results.csv.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=repository_root / "data" / "medtrace_synthetic.sqlite3",
        help="SQLite database output path.",
    )
    arguments = parser.parse_args()

    result = run_pipeline(arguments.data_dir, arguments.database)
    output = {
        "loaded_record_counts": result.loaded_record_counts,
        "validation_issue_count": len(result.validation_report.issues),
        "validation_issues": result.validation_report.to_list(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
