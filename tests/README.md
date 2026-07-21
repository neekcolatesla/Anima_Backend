# Anima — system tests

Automated tests for the Anima backend, organised by subsystem:

```
tests/
├── conftest.py            # app import path + an in-memory MOCK DATABASE + shared fixtures
├── pytest.ini             # marker registration
├── requirements-test.txt  # pytest + httpx (the rest come from the app's requirements.txt)
├── ingestion/             # ingestion workflow  (this milestone)
├── mri_analysis/          # (to come)
├── dsm5_analysis/         # (to come)
├── combined_analysis/     # (to come)
└── xai/                   # (to come)
```

## Two kinds of test (markers)

- **`unit`** — fast, offline. They run the *real* ingestion code against an
  in-memory **mock database** (`conftest.MockDB`), so no SQL Server, scans, or
  models are needed. This is the "add mock data into the mock dataset" layer.
- **`live`** — integration tests that hit the **real** SQL Server. They auto-skip
  if the database is unreachable, so they never break an offline run. They *do*
  write to the database (idempotently).

## Running

From the repo root (or from `tests/`):

```powershell
pip install -r tests/requirements-test.txt        # once

pytest tests/ingestion                # unit tests only; live tests skipped by default
pytest tests/ingestion -m unit        # same - only the fast offline unit tests
pytest tests/ingestion -m live        # opt IN to the live DB integration test
```

## What ingestion covers

- **`test_csv_ingestion.py`** (unit) — the phenotypic + DSM-5 CSV endpoints via
  FastAPI's `TestClient` with the DB mocked: 7-digit ID normalisation, `-999`/`N/A`
  → NULL, `is_child` from age, DX/score mapping, **name encryption at rest**,
  idempotent re-ingest, missing-column / empty-file rejection, and note enrichment.
- **`test_mri_ingestion.py`** (unit) — the MRI core against synthetic NIfTI volumes
  (output redirected to a temp folder): anat/anat_gm location, slice→JPEG stack,
  the longitudinal `replace` vs `new_session` behaviour, session resolution, and
  the `PatientNotFound` / `ScanNotFound` / invalid-mode errors.
- **`test_seed_live.py`** (live) — runs the real **`seed_dsm5.py`** for patients
  **130–222** against your database, then verifies the correct tables were updated:
  the new patients land in `Patient` **and** `DSM5_Assessment` (in lock-step) with
  notes attached, `MRI` / `Analysis_Result` are untouched, a sample of patients has
  the right demographics/DX, and re-running inserts no duplicates.

  > This test writes ~93 patients to the live DB (that's the point). It's
  > idempotent, so safe to re-run. It needs the DB up, the schema already created,
  > `.env` configured, and both `data/*_130-222.csv` files present.
