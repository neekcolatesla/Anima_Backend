# Anima — MS SQL Server setup (Windows + Docker Desktop)

Stand up a populated Anima database so you can build and test the DSM-5 training
script. SQL Server runs in a Docker container; you run Python from your `.venv`
on Windows and connect to it on `localhost:1433`.

You do **not** need to learn Linux — every command below runs in Windows
PowerShell / VS Code. Docker manages the Linux container for you.

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
│   ├── MRI_Ingestion.py
│   ├── MRI_Analysis.py
│   ├── DSM5_Assessment.py
│   ├── DSM5_Analysis.py
│   ├── dsm5_features.py
│   └── seed_db.py       # <- run this to create + populate the database
├── db/
│   └── init_db.sql      # schema + seed admin (single source of truth)
├── data/                # seed CSVs (git-ignored)
│   ├── NYU_Athena_Phenotypic_1-129.csv
│   └── DSM5_data_1-129.csv
├── scripts/
│   └── generate_dsm5_dataset.py   # dev tooling (synthetic DSM-5 generator)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
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
run `seed_db.py` from inside `app\`, so you don't need a second copy.

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
python seed_db.py
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
existing patients are skipped, assessments are enriched in place).

To seed a different set (e.g. the full set, or the held-out block), point it at
other files:

```powershell
python seed_db.py --phenotypic ..\data\NYU_Athena_Phenotypic_All.csv --dsm5 ..\data\DSM5_data.csv
```

Useful flags:

- `--init-sql ..\db\init_db.sql` — override the schema script location.
- `--skip-schema` — data only, assume the schema already exists.

> If your filenames differ (capitalisation, `.csv` extension), pass the exact
> paths with `--phenotypic` / `--dsm5`.

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
docker compose run --rm -v ${PWD}:/repo -w /app api python seed_db.py `
  --phenotypic /repo/data/NYU_Athena_Phenotypic_1-129.csv `
  --dsm5 /repo/data/DSM5_data_1-129.csv `
  --init-sql /repo/db/init_db.sql
```

(Set `DB_HOST` back to `localhost` afterwards if you run Python from the host.)
