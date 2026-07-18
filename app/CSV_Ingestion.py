"""
Anima - CSV ingestion pipeline.

Owns the /api/ingest/csv endpoint: bulk-ingests the NYU Athena Phenotypic CSV
into the Patient + DSM5_Assessment tables. Names are encrypted at rest with
Fernet; -999/N/A become SQL NULL; ScanDir ID is normalised to a 7-digit
patient_ID. (SysArchitecture: "api/ingest - Data Pipelines".)
"""

import io
import json
import math
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

import pyodbc
import pandas as pd

from database import get_db
from security import encrypt_value
from DSM5_Assessment import latest_assessment_id, patient_exists

logger = logging.getLogger("anima.ingest.csv")

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

# Columns expected in a DSM-5 bulk CSV (as produced by generate_dsm5_dataset.py).
DSM5_CSV_COLUMNS = {"patient_ID", "raw_answers_json", "clinician_notes"}

# CSV (NYU Athena Phenotypic) column  ->  DSM5_Assessment column
CSV_TO_DSM5 = {
    "DX": "ground_truth_dx",
    "ADHD Index": "adhd_index",
    "Inattentive": "inattentive_score",
    "Hyper/Impulsive": "hyperactive_score",
    "IQ Measure": "iq_measure",
    "Med Status": "med_status",
}
# Columns the ingest pipeline needs present in the upload.
REQUIRED_CSV_COLUMNS = {"ScanDir ID", "Age", "Gender", "Name", *CSV_TO_DSM5.keys()}
# Sentinel values in the source data that represent "missing" -> SQL NULL.
NULL_TOKENS = {"-999", "N/A", "", "NA", "NAN", "NONE"}


def _clean(value) -> Optional[str]:
    """Trim a cell and map missing/sentinel values (-999, N/A, blank) to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    return None if s.upper() in NULL_TOKENS else s


def _to_int(value) -> Optional[int]:
    s = _clean(value)
    if s is None:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _to_decimal(value) -> Optional[float]:
    s = _clean(value)
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


@router.post("/csv")
async def ingest_csv(file: UploadFile = File(...),
                     conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Bulk-ingest the NYU Athena Phenotypic CSV into Patient + DSM5_Assessment.

    Names are encrypted at rest with Fernet; -999/N/A become SQL NULL; ScanDir ID
    is normalised to a strict 7-digit patient_ID. Existing patient_IDs are skipped
    so the load is idempotent. Runs as a single transaction.
    """
    # --- Read the upload into a DataFrame ------------------------------------
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        # dtype=str + keep_default_na=False: read every cell verbatim as text.
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required column(s): {sorted(missing)}",
        )

    # Step 1: replace '-999' and 'N/A' strings with Python None (SQL NULL).
    df = df.replace({"-999": None, "N/A": None})

    ingested = 0
    skipped_existing = 0
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            # Step 2: strict 7-digit patient_ID (preserve leading zeros).
            raw_id = _clean(row["ScanDir ID"])
            if raw_id is None:
                continue  # cannot key a patient without a ScanDir ID
            patient_id = raw_id.split(".")[0].zfill(7)

            # Step 3: age + dynamic is_child flag (< 18 => child).
            age = _to_decimal(row["Age"])
            is_child = 1 if (age is not None and age < 18) else 0

            # Step 4: Gender -> biological_sex (1 = Male, 0 = Female).
            biological_sex = _to_int(row["Gender"])

            # Step 5: encrypt the plaintext Name with Fernet (encrypted at rest).
            name_plain = _clean(row["Name"])
            name_encrypted = encrypt_value(name_plain) if name_plain else None

            # DB step 2: skip patients that already exist (idempotent load).
            cursor.execute(
                "SELECT 1 FROM dbo.Patient WHERE patient_ID = ?;", patient_id
            )
            if cursor.fetchone():
                skipped_existing += 1
                continue

            # DB step 3: insert Patient (user_ID and guardian_ID left NULL here).
            cursor.execute(
                """
                INSERT INTO dbo.Patient
                    (patient_ID, user_ID, guardian_ID, name, biological_sex, is_child, age)
                VALUES (?, NULL, NULL, ?, ?, ?, ?);
                """,
                patient_id, name_encrypted, biological_sex, is_child, age,
            )

            # DB step 4: insert the corresponding DSM5_Assessment clinical row.
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
            ingested += 1

        # DB step 5: commit the whole batch as one transaction.
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("CSV ingestion DB error.")
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}")
    except Exception:
        conn.rollback()
        logger.exception("Unexpected CSV ingestion error.")
        raise HTTPException(status_code=500, detail="Internal ingestion error.")

    # PowerApps-friendly JSON payload.
    return {
        "status": "success",
        "message": f"Ingested {ingested} patient record(s).",
        "ingested_count": ingested,
        "skipped_existing": skipped_existing,
        "total_rows": int(len(df)),
    }


@router.post("/dsm5-csv")
async def ingest_dsm5_csv(file: UploadFile = File(...),
                          conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Bulk-ingest a DSM-5 CSV (patient_ID, raw_answers_json, clinician_notes).

    Matches the file produced by generate_dsm5_dataset.py. For each row the
    patient's latest DSM5_Assessment is enriched with the questionnaire JSON and
    narrative (or a new row is inserted if the patient has none). Rows whose
    patient does not exist, or whose raw_answers_json is not valid JSON, are
    skipped and reported. Runs as a single transaction.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    missing = DSM5_CSV_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"DSM-5 CSV is missing required column(s): {sorted(missing)}",
        )

    updated = inserted = 0
    skipped_missing_patient = 0
    skipped_bad_json = 0
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            patient_id = str(row["patient_ID"]).strip()
            if not patient_id:
                continue

            # FK integrity: the patient must exist before attaching an assessment.
            if not patient_exists(cursor, patient_id):
                skipped_missing_patient += 1
                continue

            raw_json = str(row["raw_answers_json"]).strip() or None
            if raw_json is not None:
                try:
                    json.loads(raw_json)  # validate before the DB ISJSON CHECK
                except (ValueError, TypeError):
                    skipped_bad_json += 1
                    continue

            notes = str(row["clinician_notes"]).strip() or None

            # Enrich the patient's latest assessment, or insert one if none.
            aid = latest_assessment_id(cursor, patient_id)
            if aid is not None:
                cursor.execute(
                    """
                    UPDATE dbo.DSM5_Assessment
                    SET raw_answers = ?, clinician_notes = ?
                    WHERE assessment_ID = ?;
                    """,
                    raw_json, notes, aid,
                )
                updated += 1
            else:
                cursor.execute(
                    """
                    INSERT INTO dbo.DSM5_Assessment (patient_ID, raw_answers, clinician_notes)
                    VALUES (?, ?, ?);
                    """,
                    patient_id, raw_json, notes,
                )
                inserted += 1

        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("DSM-5 CSV ingestion DB error.")
        raise HTTPException(status_code=400, detail=f"DSM-5 ingestion failed: {exc}")
    except Exception:
        conn.rollback()
        logger.exception("Unexpected DSM-5 CSV ingestion error.")
        raise HTTPException(status_code=500, detail="Internal DSM-5 ingestion error.")

    return {
        "status": "success",
        "message": (f"DSM-5 data applied: {updated} updated, {inserted} inserted."),
        "updated_count": updated,
        "inserted_count": inserted,
        "skipped_missing_patient": skipped_missing_patient,
        "skipped_invalid_json": skipped_bad_json,
        "total_rows": int(len(df)),
    }