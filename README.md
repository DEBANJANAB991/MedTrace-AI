# MedTrace AI

> **Evidence-grounded healthcare intelligence for more trustworthy decisions.**

MedTrace AI is a hackathon prototype for **Challenge 3 — Exasol: Healthcare AI You Can Trust** at Healthcare Hackathon Bayern, Erlangen.

Healthcare information is often fragmented across clinical notes, medication records, laboratory results and other systems. MedTrace AI explores how these records can be brought together to surface useful insights while making supporting evidence, uncertainty and data-quality limitations visible.

**This is an experimental prototype. It is not a medical device, diagnostic system or substitute for professional clinical judgement. Use synthetic healthcare data only.**

## The problem

Healthcare data can be distributed across systems, incomplete, out of date, inconsistent or difficult to interpret. A generic chatbot does not solve these problems. A useful healthcare AI application must show where findings come from, identify gaps and contradictions, and communicate when evidence is insufficient.

## Our approach

MedTrace AI is designed around four principles:

1. **Connect** relevant structured and unstructured healthcare records.
2. **Validate** missing fields, duplicates, conflicting values and temporal inconsistencies.
3. **Explain** findings using retrieved source records and traceable references.
4. **Keep humans in control** by showing evidence and uncertainty for human review.

### Intended demonstration

A user asks:

> "Summarise the available history for this patient. Which records appear inconsistent, and what information is missing?"

The prototype should retrieve relevant records, display source and date, flag data-quality issues, generate an evidence-linked explanation, state what cannot be concluded, and mark the output for human review. It does not generate diagnoses or treatment recommendations.

## Trust and safety principles

| Principle | Prototype behaviour |
|---|---|
| Evidence traceability | Factual findings link to source record IDs. |
| Uncertainty | Insufficient or ambiguous evidence is explicitly reported. |
| Data quality | Missing, duplicate and conflicting records are surfaced. |
| Human oversight | Outputs are informational and require human review. |
| Privacy | Use synthetic or properly authorised, de-identified data only. |
| Separation of concerns | Deterministic checks/calculations run in code or SQL, not delegated to an LLM. |

## Architecture

```mermaid
flowchart TD
    A["Synthetic / authorised healthcare data"] --> B["Ingestion and normalisation"]
    B --> C["Data validation and quality checks"]
    C --> D["Analytics and query layer"]
    D --> E["Evidence retrieval"]
    E --> F["LLM explanation layer"]
    F --> G["Evidence-linked insights"]
    G --> H["Human review"]
    C --> G
```

## Planned technology

| Layer | Initial direction | Possible hackathon extension |
|---|---|---|
| Language | Python | Python |
| Ingestion | CSV / pandas | Additional formats/connectors |
| Storage | SQLite for local prototype | Exasol Personal / team-selected database |
| Validation | Python and SQL rules | More comprehensive quality checks |
| Retrieval | Transparent keyword baseline | Embeddings / vector retrieval |
| AI | Optional LLM API | Team-selected model and orchestration |
| Interface | CLI initially | Streamlit or React |
| Testing | pytest | Scenario-based evaluation |

The stack is intentionally flexible. Exasol Personal is a potential analytics foundation for structured healthcare data; the final integration depends on the team's chosen solution and available access.

## Repository structure

```text
medtrace-ai/
├── data/
│   ├── synthetic/       # Fictional CSV demonstration data only
│   └── medtrace_synthetic.sqlite3
├── docs/                # Architecture and decision notes
├── src/
│   └── medtrace/        # Ingestion, validation, models, and SQLite modules
├── tests/               # Unit and pipeline tests
├── pyproject.toml       # Python 3.12 package and pytest configuration
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting started

### Requirements
- Python 3.12
- Git
- VS Code or another Python IDE

### 1. Clone
```bash
git clone https://github.com/<your-github-username>/medtrace-ai.git
cd medtrace-ai
```

### 2. Create a virtual environment

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Configure environment
Copy `.env.example` to `.env` and add only credentials required by the components you actually use. Never commit `.env` or API keys.

### 5. Run
Run the ingestion and validation pipeline from the repository root:

```bash
python -m medtrace
```

By default, the command reads `data/synthetic/patients.csv`,
`data/synthetic/medications.csv`, and
`data/synthetic/laboratory_results.csv`. It writes the validated snapshot to
`data/medtrace_synthetic.sqlite3` and prints a JSON validation report. Override
the paths when needed:

```bash
python -m medtrace --data-dir data/synthetic --database data/demo.sqlite3
```

The CSVs intentionally contain missing values, repeated record IDs, invalid
dates, overlapping medication records with contradictory statuses, and unknown
patient references. These rows appear in the report and are not inserted into
SQLite. Valid records are retained, and foreign keys prevent orphan medication
or laboratory rows. An empty medication `end_date` is accepted only for an
`active` medication, where it represents an open-ended interval.

Run the tests with:

```bash
python -m pytest
```

### Database queries

The tables are `patients`, `medications`, and `laboratory_results`. For example,
join a laboratory result to its patient while retaining the source and
observation date:

```sql
SELECT
    patient.patient_id,
    result.result_id,
    result.test_name,
    result.value,
    result.unit,
    result.observation_date,
    result.source
FROM laboratory_results AS result
JOIN patients AS patient USING (patient_id)
ORDER BY patient.patient_id, result.observation_date;
```

To check medication references, join medications to patients:

```sql
SELECT
    medication.medication_id,
    patient.patient_id,
    medication.medication_name,
    medication.status,
    medication.source
FROM medications AS medication
JOIN patients AS patient USING (patient_id);
```

SQLite foreign-key enforcement is enabled on every application connection;
`PRAGMA foreign_key_check;` can be used to verify the stored database.

### Implementation notes

- `ingestion.py` reads CSVs with pandas and carries the original CSV line number
  into each in-memory row so validation findings point back to their source.
- `validation.py` performs deterministic checks before persistence. A
  medication-status conflict means records for the same patient and medication
  have different statuses during overlapping date intervals. An active record
  with no end date has an open-ended interval.
- `database.py` owns SQLite connections, schema creation, transactions, and
  parameterized inserts. The patient table is the parent table; medication and
  laboratory tables reference it with foreign keys.
- `pipeline.py` connects the stages. Any row with an error is excluded from the
  database snapshot, while its structured issue remains in the JSON report.
  The snapshot is replaced as a transaction, and database failures are raised
  rather than converted into a success-shaped result.
- `models.py` defines typed issue and pipeline results. No RAG, embeddings,
  LLM, or user interface is included in this stage.

## Evaluation plan

Use synthetic test scenarios to evaluate:
- Whether the correct supporting records are retrieved.
- Whether findings link to source IDs.
- Whether deliberately introduced missing/conflicting data is detected.
- Whether unsupported claims are avoided.
- Whether insufficient evidence is clearly communicated.
- Whether a human reviewer can understand why a finding was produced.

A strong demo should show both a supported finding and a case where the system correctly declines to draw a conclusion.

## Roadmap

- [ ] Confirm target user and exact problem with the hackathon team.
- [ ] Define the MVP and data model.
- [ ] Implement ingestion and normalisation.
- [ ] Add data-quality and conflict checks.
- [ ] Integrate Exasol if selected and available.
- [ ] Implement evidence retrieval.
- [ ] Add an LLM explanation layer with source references.
- [ ] Build a simple interface.
- [ ] Test missing, conflicting and insufficient-evidence scenarios.
- [ ] Prepare a reproducible end-to-end demo.

## Team

Add final team members and roles here.

## Acknowledgement

Built for **Challenge 3 — Exasol: Healthcare AI You Can Trust**, Healthcare Hackathon Bayern, Medical Valley EMN.

Challenge brief: https://hackathon.medical-valley-emn.de/healthcare-ai-you-can-trust/

## Disclaimer

This project is an experimental hackathon prototype for educational and demonstration purposes. It does not provide medical advice, diagnosis, triage or treatment recommendations. All outputs require appropriate human review. Use synthetic healthcare data only.
