# MedTrace AI Project Guidelines

## Project goals and safety

- Build a trustworthy healthcare AI hackathon prototype using synthetic healthcare data only. Do not introduce real patient data.
- This project is informational, not diagnostic. Never generate diagnoses, triage decisions, or treatment recommendations.
- Every factual claim in AI-generated output must be supported by retrieved evidence. Preserve source IDs, timestamps, and provenance from ingestion through retrieval and output so evidence can be cited.
- Communicate uncertainty, missing evidence, and conflicting records explicitly. Do not fill gaps with assumptions or present unsupported claims as facts.
- Keep deterministic validation, analytics, and calculations in Python or SQL; use the LLM only to explain supplied evidence.
- Keep human review in the workflow for generated healthcare summaries.

## Python and architecture

- Target Python 3.12.
- Use clear, meaningful names and small, focused modules. Keep ingestion, database access, validation, retrieval, and LLM integration as separate responsibilities.
- Prefer existing project patterns and the simplest implementation that meets the requirement. Avoid unnecessary dependencies, abstractions, and overengineering.
- Handle failures explicitly; do not silently discard errors or substitute unsupported defaults.

## Tests and documentation

- Add or update pytest tests for important behavior, especially data validation, provenance, evidence support, uncertainty, and insufficient-evidence cases.
- Run the most focused relevant tests, then broader checks when warranted. Do not claim tests pass unless they were run.
- Update the README when user-visible functionality, setup, or run instructions change.
- Explain non-obvious implementation decisions and unfamiliar code in the response so the project remains approachable to someone learning LLM application development.
