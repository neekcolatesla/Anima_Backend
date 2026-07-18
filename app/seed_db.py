"""
Anima - database seeding (dev / evaluation helper).

One command to get a populated Anima database for developing and testing the
DSM-5 training script (and demos):

  1. creates the schema + seed administrator by running db/init_db.sql,
  2. loads the NYU Athena Phenotypic CSV -> Patient + DSM5_Assessment
     (demographics, DX label, subscale scores; names encrypted at rest),
  3. loads the DSM-5 CSV -> raw_answers + clinician_notes on those assessments.

Re-uses the exact cleaning / mapping / encryption helpers the API ingestion
endpoints use, so the seeded rows are identical to a real ingest. Idempotent:
existing patients are skipped and existing assessments are enriched in place
(init_db.sql is itself idempotent, so re-running the schema step is safe too).

Run (from the app/ folder, with the SQL Server container up and .env set):
    python seed_db.py
"""

import os
import re
import sys
import json
import argparse
import logging

import pandas as pd

from database import get_connection
from security import encrypt_value
from CSV_Ingestion import _clean, _to_int, _to_decimal
from DSM5_Assessment import latest_assessment_id, patient_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anima.seed")

# seed_db.py lives in app/, so the repo root is two levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Schema + seed-admin script (single source of truth; no init_db.py).
DEFAULT_INIT_SQL = os.path.join(_REPO_ROOT, "db", "init_db.sql")

# The data is pre-split into a "seed" set (patients 1-129) and a held-out set
# (130-222) that gets ingested live during the moderated sessions. seed_db loads
# the 1-129 files by default; point --phenotypic / --dsm5 at other files (e.g.
# the *_130-222 or *_All versions) to seed a different set.
DEFAULT_PHENOTYPIC = os.path.join(_REPO_ROOT, "data", "NYU_Athena_Phenotypic_1-129.csv")
DEFAULT_DSM5 = os.path.join(_REPO_ROOT, "data", "DSM5_data_1-129.csv")


# =============================================================================
# Schema creation (run init_db.sql instead of importing a Python init module)
# =============================================================================
def _split_go_batches(sql_text: str) -> list:
    """Split a T-SQL script into batches on the GO separator.

    GO is a client-side batch separator (not a T-SQL statement), so pyodbc
    cannot execute it - the script must be sent one batch at a time. Matches a
    line that is only GO (any case, optional surrounding whitespace).
    """
    parts = re.split(r"(?im)^\s*GO\s*$", sql_text)
    return [p.strip() for p in parts if p.strip()]


def init_schema(sql_path: str) -> None:
    """Create the schema + seed admin by executing db/init_db.sql.

    CREATE DATABASE / USE cannot run inside a multi-statement transaction, so we
    connect to ``master`` with autocommit ON and run each GO-separated batch on
    its own. init_db.sql is idempotent (guarded by IF OBJECT_ID / IF NOT EXISTS),
    so this is safe to re-run.
    """
    with open(sql_path, "r", encoding="utf-8") as fh:
        sql_text = fh.read()

    batches = _split_go_batches(sql_text)
    logger.info("Executing %s SQL batch(es) from %s ...", len(batches), sql_path)

    conn = get_connection(database="master")
    conn.autocommit = True   # required for CREATE DATABASE / USE
    try:
        cursor = conn.cursor()
        for i, batch in enumerate(batches, start=1):
            try:
                cursor.execute(batch)
            except Exception as exc:
                raise RuntimeError(f"init_db.sql batch #{i} failed: {exc}") from exc
    finally:
        conn.close()


# =============================================================================
# Loaders (mirror the API ingestion so seeded data == ingested data)
# =============================================================================
def load_phenotypic(cursor, csv_path: str) -> dict:
    """Load demographics + DX + subscale scores into Patient + DSM5_Assessment."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df.replace({"-999": None, "N/A": None})

    inserted = 0
    skipped = 0
    for _, row in df.iterrows():
        raw_id = _clean(row["ScanDir ID"])
        if raw_id is None:
            continue

        patient_id = raw_id.split(".")[0].zfill(7)

        cursor.execute("SELECT 1 FROM dbo.Patient WHERE patient_ID = ?;", patient_id)
        if cursor.fetchone():
            skipped += 1
            continue

        age = _to_decimal(row["Age"])
        is_child = 1 if (age is not None and age < 18) else 0
        biological_sex = _to_int(row["Gender"])
        name_plain = _clean(row["Name"]) if "Name" in df.columns else None
        name_encrypted = encrypt_value(name_plain) if name_plain else None

        cursor.execute(
            """
            INSERT INTO dbo.Patient
                (patient_ID, user_ID, guardian_ID, name, biological_sex, is_child, age)
            VALUES (?, NULL, NULL, ?, ?, ?, ?);
            """,
            patient_id, name_encrypted, biological_sex, is_child, age,
        )
        cursor.execute(
            """
            INSERT INTO dbo.DSM5_Assessment
                (patient_ID, ground_truth_dx, adhd_index, inattentive_score,
                 hyperactive_score, iq_measure, med_status)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            patient_id,
            _to_int(row["DX"]),
            _to_int(row["ADHD Index"]),
            _to_int(row["Inattentive"]),
            _to_int(row["Hyper/Impulsive"]),
            _to_int(row["IQ Measure"]),
            _to_int(row["Med Status"]),
        )
        inserted += 1

    return {"patients_inserted": inserted, "patients_skipped": skipped,
            "rows": int(len(df))}


def load_dsm5(cursor, csv_path: str) -> dict:
    """Enrich each patient's latest DSM5_Assessment with raw_answers + notes."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    updated = 0
    skipped = 0
    for _, row in df.iterrows():
        patient_id = str(row["patient_ID"]).strip()

        if not patient_id or not patient_exists(cursor, patient_id):
            skipped += 1
            continue

        raw_json = str(row["raw_answers_json"]).strip() or None
        if raw_json is not None:
            try:
                json.loads(raw_json)
            except (ValueError, TypeError):
                skipped += 1
                continue
        notes = str(row["clinician_notes"]).strip() or None

        aid = latest_assessment_id(cursor, patient_id)
        if aid is not None:
            cursor.execute(
                "UPDATE dbo.DSM5_Assessment SET raw_answers = ?, clinician_notes = ? "
                "WHERE assessment_ID = ?;",
                raw_json, notes, aid,
            )
        else:
            cursor.execute(
                "INSERT INTO dbo.DSM5_Assessment (patient_ID, raw_answers, clinician_notes) "
                "VALUES (?, ?, ?);",
                patient_id, raw_json, notes,
            )
        updated += 1

    return {"assessments_enriched": updated, "skipped": skipped, "rows": int(len(df))}


def _counts(cursor) -> dict:
    out = {}
    for table in ("Patient", "DSM5_Assessment"):
        cursor.execute(f"SELECT COUNT(*) FROM dbo.{table};")
        out[table] = cursor.fetchone()[0]
    cursor.execute(
        "SELECT COUNT(*) FROM dbo.DSM5_Assessment WHERE clinician_notes IS NOT NULL;"
    )
    out["assessments_with_notes"] = cursor.fetchone()[0]
    return out


# =============================================================================
# Entrypoint
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the Anima database.")
    ap.add_argument("--phenotypic", default=DEFAULT_PHENOTYPIC,
                    help="NYU Athena Phenotypic CSV (with a Name column).")
    ap.add_argument("--dsm5", default=DEFAULT_DSM5,
                    help="DSM-5 CSV (patient_ID, raw_answers_json, clinician_notes).")
    ap.add_argument("--init-sql", default=DEFAULT_INIT_SQL,
                    help="Path to the schema script (db/init_db.sql).")
    ap.add_argument("--skip-schema", action="store_true",
                    help="Skip schema creation (assume it already exists).")
    args = ap.parse_args()

    required = [args.phenotypic, args.dsm5]
    if not args.skip_schema:
        required.append(args.init_sql)
    for path in required:
        if not os.path.exists(path):
            sys.exit(f"ERROR: file not found: {path}")

    # 1. Schema + seed admin (idempotent) by running init_db.sql.
    if not args.skip_schema:
        logger.info("Creating schema + seed administrator from %s ...", args.init_sql)
        try:
            init_schema(args.init_sql)
        except Exception as exc:
            sys.exit(f"ERROR: schema initialisation failed ({exc}). Is SQL Server up?")

    # 2 + 3. Load data inside one transaction.
    conn = get_connection()
    try:
        cursor = conn.cursor()
        pheno = load_phenotypic(cursor, args.phenotypic)
        dsm5 = load_dsm5(cursor, args.dsm5)
        conn.commit()
        totals = _counts(cursor)
    except Exception:
        conn.rollback()
        logger.exception("Seeding failed - rolled back.")
        return 1
    finally:
        conn.close()

    logger.info("Phenotypic load: %s", pheno)
    logger.info("DSM-5 load:      %s", dsm5)
    logger.info("DB totals:       %s", totals)
    print("\nSeed complete.")
    print(f"  Source files:             {os.path.basename(args.phenotypic)}, "
          f"{os.path.basename(args.dsm5)}")
    print(f"  Patients:                 {totals['Patient']}")
    print(f"  DSM5 assessments:         {totals['DSM5_Assessment']}")
    print(f"  ...with clinician_notes:  {totals['assessments_with_notes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
