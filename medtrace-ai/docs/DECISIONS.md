# Architecture decisions

## Initial direction
Start with a narrow, demonstrable workflow and synthetic data. Keep the repository modular until the team confirms its user, problem and data source.

## Trust boundary
Use SQL/Python for deterministic validation and calculations. The LLM may summarise retrieved evidence but must not invent missing facts or make clinical decisions.

## Storage
SQLite is suitable for local development. Evaluate Exasol Personal for the hackathon analytics layer if it fits the selected use case and access is available.
