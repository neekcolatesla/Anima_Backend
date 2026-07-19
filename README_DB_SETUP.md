# Anima — MS SQL Server setup (Windows + Docker Desktop)

Stand up a populated Anima database so you can build and test the DSM-5 training
script. SQL Server runs in a Docker container; you run Python from your `.venv`
on Windows and connect to it on `localhost:1433`.

You do **not** need to learn Linux — every command below runs in Windows
PowerShell / VS Code. Docker manages the Linux container for you.

---

## Quick start: whole stack in one command (auto-seed)

For evaluators, the fastest path is to let Docker do everything. Two prerequisites
first, both from the repo root:

- A `.env` file (copy `.env.example` → `.env` and fill it in — see §2).
- The 1-129 seed CSVs present in `data/`: `NYU_Athena_Phenotypic_1-129.csv` and
  `DSM5_data_1-129.csv`. **These are not in the git repo** — the ADHD-200 source
  data is kept out of version control, so the files are provided with the project
  folder/zip. If you only have a bare clone, place them in `data/` before running.

Then:

```powershell
docker compose up
```

This starts three services in order:

1. **db** — SQL Server, waits until it reports healthy;
2. **seed** — a one-shot container that runs `db/init_db.sql` (schema + seed
   admin) and loads the 1-129 patients from `data/`, then exits;
3. **api** — FastAPI on <http://localhost:8000> (and `/docs`), started only once
   the seed has completed successfully.

When it finishes you have a populated database **and** a running API with no
manual steps. The seeder is idempotent, so re-running `docker compose up` is
safe — it skips patients that already exist. The held-out 130-222 set is never
loaded here.

> Notes: Bio_ClinicalBERT is **baked into the API image** at build time, so the
> running API needs no network — the ~440 MB model download happens once during
> `docker compose build` (which needs internet), not at request time. The trained
> model `app/models/dsm5_head.pt` must be present for real (non-heuristic)
> predictions; it ships with the project folder.

The rest of this document is the **manual / development** path: run SQL Server in
Docker but drive seeding and training from your Windows `.venv`.

---

## The API image (offline model bake)

The API's text model uses Bio_ClinicalBERT. Rather than downloading it from
Hugging Face on the first request, the `Dockerfile` **bakes it into the image** so
the running container is fully self-contained.

- **At build time** (`docker compose build` / `up --build`): the Dockerfile runs
  `scripts/predownload_model.py`, which downloads the model (~440 MB) into the
  image's Hugging Face cache at `HF_HOME=/opt/hf-cache`. This needs internet
  **once**, and is cached as its own image layer — later builds skip it unless the
  Dockerfile or that script changes. (It is placed before the app `COPY`, so
  editing application code doesn't invalidate the expensive download layer.)
- **At run time**: the image sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`,
  so the API loads the model straight from the baked cache with **no network
  calls** — it works even air-gapped. The `./app:/app` dev mount doesn't disturb
  this, since the cache lives outside `/app`.

Practical implications:

- The image is large (PyTorch + the baked model, several GB) — expected.
- The `seed` and `api` services share one image; the model layer is
  content-addressed, so it is stored once, not duplicated.
- To confirm the offline load, score a patient and watch the `anima_api` logs:
  `Loading clinical language model ...` → `Clinical language model loaded.` with
  **no** `huggingface.co` request lines between them, and `scoring_method: model`
  in the response.
- To point at a different model, set `CLINICAL_BERT_MODEL` and rebuild with
  `--build` so the new model is baked in.

See `README_DSM5.md` → "Pre-downloading the model & offline mode" for the
host (`.venv`) equivalent used by the training and test scripts.

---

## 0. Repo layout

```
Anima_Backend/
├── app/                 # FastAPI application (mounted into the container as /app)
│   ├── main.py
│   ├── database.py
│   ├── security.py
│   ├── Auth_RBAC.py
│   ├── CSV_Ingestion.py
│   ├── MRI_Ingestion.py       # /api/ingest/mri + shared longitudinal core
│   ├── MRI_Analysis.py
│   ├── DSM5_Assessment.py
│   ├── DSM5_Analysis.py
│   ├── dsm5_features.py       # shared BERT + demographic feature builder
│   ├── dsm5_model.py          # trained head architecture (shared train/serve)
│   ├── train_dsm5.py          # DSM-5 model training (see README_DSM5.md)
│   ├── dsm5_smoketest.py       # smoke-test the analysis endpoint
│   ├── seed_dsm5.py             # <- schema + demographics/DSM-5 data
│   ├── seed_mri.py            # <- bulk-load MRI training scans (§4a)
│   └── models/
│       └── dsm5_head.pt       # trained weights (ships with the folder)
├── db/
│   └── init_db.sql      # schema + seed admin (single source of truth)
├── data/                # seed CSVs (git-ignored, delivered with the folder)
│   ├── NYU_Athena_Phenotypic_1-129.csv
│   ├── DSM5_data_1-129.csv
│   └── mri/             # raw MRI scans (git-ignored; NOT committed)
│       └── NYU_Athena_preproc_1-129/<patient>/*.nii.gz
├── scripts/
│   ├── generate_dsm5_dataset.py   # dev tooling (synthetic DSM-5 generator)
│   └── predownload_model.py       # cache Bio_ClinicalBERT (offline / image bake)
├── docker-compose.yml
├── Dockerfile                     # bakes Bio_ClinicalBERT for offline runtime
├── requirements.txt
├── README_DB_SETUP.md
├── README_DSM5.md
├── LIMITATIONS.md
├── .env                 # your local secrets (git-ignored)
└── .env.example
```

Everything below is run from the **repo root** unless it says otherwise.

---

## 1. Prerequisites

- **Docker Desktop** for Windows, installed and running.
- Your **`.venv`** with `requirements.txt` installed (`pyodbc` in particular).
- The two seed CSVs in `data/`:
  - `data\NYU_Athena_Phenotypic_1-129.csv`
  - `data\DSM5_data_1-129.csv`
- **Microsoft ODBC Driver 18 for SQL Server** installed on Windows — `pyodbc`
  needs it to connect. One-time MSI from Microsoft:
  <https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server>

---

## 2. Configure `.env`

Copy the example and fill it in (keep `.env` out of git):

```powershell
copy .env.example .env
```

Set these values in `.env`:

```
# SQL Server SA password — needs upper+lower+digit/symbol, 8+ chars
MSSQL_SA_PASSWORD=Change_Me_Str0ng!
DB_NAME=anima
DB_USER=sa
DB_PASSWORD=Change_Me_Str0ng!          # same as MSSQL_SA_PASSWORD for local dev

# IMPORTANT: host vs container
#   DB_HOST=localhost  -> when you run Python from your Windows .venv
#   DB_HOST=db         -> when code runs INSIDE the api container
DB_HOST=localhost
DB_PORT=1433

# Field-encryption key for patient names — generate one:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FERNET_KEY=PASTE_GENERATED_KEY_HERE
```

Generate the Fernet key and paste it in:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

`.env` lives at the repo root. `database.py` finds it automatically even when you
run `seed_dsm5.py` from inside `app\`, so you don't need a second copy.

---

## 3. Start SQL Server (container)

From the repo root:

```powershell
docker compose up -d db
```

Check it's healthy (wait until `STATUS` shows `healthy`, ~20–40 s on first run):

```powershell
docker compose ps
```

---

## 4. Create the schema + load the data

From the `app` folder, with the `.venv` activated:

```powershell
cd app
python seed_dsm5.py
```

This runs `db\init_db.sql` to create the tables + seed admin (executed batch by
batch, split on `GO`), loads the phenotypic CSV into `Patient` +
`DSM5_Assessment` (encrypting names where a `Name` column exists), then loads the
DSM-5 CSV into `raw_answers` + `clinician_notes`.

By default it reads the **seed set** files (patients 1–129) from `..\data\`:
`NYU_Athena_Phenotypic_1-129.csv` and `DSM5_data_1-129.csv`. The `*_130-222`
files are your **held-out set** — kept out of the repo/DB so you can ingest them
live during the moderated sessions to demonstrate the pipeline on unseen data.
Expected output:

```
Seed complete.
  Source files:             NYU_Athena_Phenotypic_1-129.csv, DSM5_data_1-129.csv
  Patients:                 129
  DSM5 assessments:         129
  ...with clinician_notes:  129
```

It's idempotent — safe to re-run (`init_db.sql` only creates missing objects,
existing patients are skipped, assessments are enriched in place). **Re-running
also applies schema migrations**: after any `init_db.sql` change, `seed_dsm5.py`
brings your existing database up to date (adding new columns/tables), so the
batch count rises — e.g. the longitudinal-MRI + audit changes take it from 15 to
**22 batches**. Run it once after pulling schema changes and before `seed_mri`.

To seed a different set (e.g. the full set, or the held-out block), point it at
other files:

```powershell
python seed_dsm5.py --phenotypic ..\data\NYU_Athena_Phenotypic_All.csv --dsm5 ..\data\DSM5_data.csv
```

Useful flags:

- `--init-sql ..\db\init_db.sql` — override the schema script location.
- `--skip-schema` — data only, assume the schema already exists.

> If your filenames differ (capitalisation, `.csv` extension), pass the exact
> paths with `--phenotypic` / `--dsm5`.

---

## 4a. Load the MRI training scans (`seed_mri`)

Bulk-loads every patient's MRI scans for training the image model, using the same
core as the live `POST /api/ingest/mri` endpoint so seeded rows match a real
ingest.

**Prerequisites:** the database is up (§3), patients are already seeded (§4), the
schema migration has been applied (re-run `seed_dsm5.py` so the `is_current` /
`scan_session` columns exist), and the scans are on disk as per-patient folders:

```
data\mri\NYU_Athena_preproc_1-129\
├── 0010001\   wssd0010001_session_1_anat.nii.gz   swssd0010001_session_1_anat_gm.nii.gz
├── 0010002\   ...
└── 0010129\   ...
```

The folder name is the `patient_ID`; each holds one `*anat.nii(.gz)` and one
`*_anat_gm.nii(.gz)`. These scans are **git-ignored** (large ADHD-200 data), so
they live in the repo folder for the seeder but are never committed.

Smoke-test with 3 patients first, then run the full set (from `app`):

```powershell
python seed_mri.py --mri-dir ..\data\mri\NYU_Athena_preproc_1-129 --limit 3
python seed_mri.py --mri-dir ..\data\mri\NYU_Athena_preproc_1-129
```

Each patient is committed in its own transaction, so one bad folder never sinks
the batch. Each patient yields **378 slices** (189 per scan × the two scans), so
the full run writes ~**48,762** JPEGs across 129 patients and is slow.

Smoke-test output (`--limit 3`):

```
  ok   0010001: 378 slices across 2 scan(s) (session 1)
  ok   0010002: 378 slices across 2 scan(s) (session 1)
  ok   0010003: 378 slices across 2 scan(s) (session 1)
MRI seed complete.
  Ingested:     3
  Skipped:      0
  Failed:       0
  Total slices: 1134
```

The full run ends with `Ingested: 129 ... Total slices: 48762`.

Useful flags:

- `--skip-existing` — skip patients that already have a current scan (resumable).
- `--limit N` — only the first N folders; `--patient 0010001` — a single patient.
- `--mode replace|new_session` — how to treat a patient that already has scans.
  `replace` (default) corrects the current scan; `new_session` keeps the old scan
  as history (`is_current = 0`) and adds the new one as current (see the
  longitudinal MRI model).

> Hitting a `localhost,1433 ... Server is not found` connection timeout means the
> database container isn't running — start it with `docker compose up -d db`.

---

## 5. Verify

Quick check straight against the container:

```powershell
docker exec -it anima_sqlserver /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "<your SA password>" -C -Q "SELECT COUNT(*) AS patients FROM anima.dbo.Patient;"
```

…or from Python in your `.venv` (from the `app` folder):

```powershell
python -c "from database import get_connection; c=get_connection(); cur=c.cursor(); cur.execute('SELECT TOP 3 patient_ID, ground_truth_dx, inattentive_score FROM dbo.DSM5_Assessment'); print(cur.fetchall()); c.close()"
```

---

## 6. Point the training script at it

The training script reads the same `.env`, so with `DB_HOST=localhost` it
connects to the container automatically — no extra config. Features come from
`Patient` + `DSM5_Assessment`; the training label is `ground_truth_dx`.

The text model uses Bio_ClinicalBERT, which downloads (~440 MB) from Hugging Face
on first use. To avoid depending on the network at startup (recommended for demos
/ locked-down machines), pre-download it once with `python scripts/predownload_model.py`
and set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in `.env`. See
`README_DSM5.md` → "Pre-downloading the model & offline mode" for details.

---

## 7. Stop / reset

```powershell
docker compose stop db          # stop (keeps data)
docker compose down             # stop + remove container (keeps the named volume)
docker compose down -v          # WIPE everything, incl. the database volume
```

---

## Alternative: seed inside the container (no host ODBC driver)

If you'd rather not install the ODBC driver on Windows, run the seed inside the
api image (which already bundles the driver). Set `DB_HOST=db` in `.env` first,
then mount the repo so the container can see the CSVs + schema script:

```powershell
docker compose run --rm -v ${PWD}:/repo -w /app api python seed_dsm5.py `
  --phenotypic /repo/data/NYU_Athena_Phenotypic_1-129.csv `
  --dsm5 /repo/data/DSM5_data_1-129.csv `
  --init-sql /repo/db/init_db.sql
```

(Set `DB_HOST` back to `localhost` afterwards if you run Python from the host.)
