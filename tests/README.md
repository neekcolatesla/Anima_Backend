# Anima — system tests

Automated tests for the Anima backend, organised by subsystem:

```
tests/
├── conftest.py            # app import path + an in-memory MOCK DATABASE + shared fixtures
├── pytest.ini             # marker registration
├── requirements-test.txt  # pytest + httpx (the rest come from the app's requirements.txt)
├── ingestion/             # CSV + MRI ingestion, and live seeding of 130–222
├── dsm5_analysis/         # text & demographic (NLP) engine
├── mri_analysis/          # image (CNN) engine
├── combined_analysis/     # fusion engine + RBAC + audit trail
└── xai/                   # the explanation layer across all three engines
```

Every category follows the same shape: `*_unit.py` (fast, mocked, offline) plus a
`*_live.py` carrying the **same 8 scenarios** over the held-out 130–222 cohort
(3 random, 1 random, 3 inattentive / hyperactive / combined, then 30 / 50 / all).

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

pytest tests/xai                      # unit tests only; live tests skipped by default
pytest tests/xai -m unit              # same - only the fast offline unit tests
pytest tests/xai -m live -s           # opt IN to the live scenarios (-s to see printouts)
pytest tests                          # the whole suite (all unit tests; live auto-skipped)
```

Swap `xai` for any category (`ingestion`, `dsm5_analysis`, `mri_analysis`,
`combined_analysis`). The live scenarios need the DB up, `.env` set, and the
130–222 cohort seeded (DSM-5) and MRI-ingested; they auto-skip otherwise.

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

## What the analysis categories cover

- **`dsm5_analysis/`** — unit tests exercise the subtype rules, the heuristic
  fallback, and the trained-head path (BERT stubbed, a real tiny head saved) end to
  end against the mock DB; the live suite reports NLP diagnosis accuracy on 130–222.
- **`mri_analysis/`** — unit tests run the real CNN over synthetic slice folders
  (fetch → score → heatmap → persist); the live suite reports the CNN's held-out
  diagnosis accuracy on the ingested 130–222 scans.
- **`combined_analysis/`** — unit tests cover the fusion maths (0.7 NLP / 0.3 MRI),
  the graceful single-channel fallbacks, the append-only `Analysis_Result` audit row
  with `created_by`, and RBAC on both the trigger and the retrieval reads; the live
  suite reports the fused accuracy alongside NLP-only and MRI-only for comparison.
- **`xai/`** — unit tests prove each explanation is well-formed in isolation: the
  DSM-5 feature-importance bars and signed per-word "push" attribution, the MRI
  Grad-CAM pipeline (slice parsing, colour map, PNG data-URI encoding, overlay), and
  that the combined engine merges both channels and degrades the image channel to an
  honest `available: False` when scans are missing/untrained. The live suite checks,
  on 130–222, that every rendered explanation is valid — ranked ~100% bars, decodable
  Grad-CAM PNGs, and influential words that are genuine tokens **from the patient's
  own note**.
