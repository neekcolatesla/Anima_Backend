"""
Anima - FastAPI application entrypoint.

Initialises the FastAPI app, configures CORS middleware, mounts the static
directory that serves processed 2D MRI slices, and establishes the initial
SQL Server connection (via pyodbc) on startup.
"""

import os
import io
import json
import math
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, Union

from fastapi import FastAPI, APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

import pyodbc
import pandas as pd

from database import get_db, wait_for_db, DB_NAME
from security import hash_password, verify_password, encrypt_value, decrypt_value

# --- Auth / RBAC configuration -----------------------------------------------
# Email domain that is automatically elevated to the Clinician role.
CLINICIAN_EMAIL_DOMAIN = "@kingston.ac.uk"
# Role -> user_ID prefix letter (RBAC identifier convention).
ROLE_PREFIX = {"Clinician": "C", "Patient": "P", "Guardian": "G", "Admin": "A"}
# Roles a user is permitted to REQUEST at registration.
SELF_REGISTERABLE_ROLES = {"Patient", "Guardian"}
# One-time passcode settings.
OTP_TTL_MINUTES = int(os.getenv("OTP_TTL_MINUTES", "10"))
ROLE_ID_WIDTH = 7  # 7-digit zero-padded numeric role IDs, e.g. '0010006'

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anima.main")

# --- Application -------------------------------------------------------------
app = FastAPI(
    title="Anima API",
    description="AI-integrated ADHD diagnosis support backend.",
    version="0.1.0",
)

# --- CORS middleware ---------------------------------------------------------
# PowerApps / Outlook clients call this API from different origins.
# Override with a comma-separated ALLOWED_ORIGINS env var in production.
_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"] if _origins_env.strip() == "*"
    else [o.strip() for o in _origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files: processed MRI slices --------------------------------------
# Matches the Docker volume mount at /app/static/mri_images.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(os.path.join(STATIC_DIR, "mri_images"), exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Lifecycle events --------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    """Block until SQL Server is reachable so the API doesn't serve traffic
    before its database dependency is available."""
    logger.info("Anima API starting up - checking SQL Server connectivity...")
    if not wait_for_db():
        # Log loudly but let the app start so /health can report the problem.
        logger.error("Startup completed WITHOUT a confirmed database connection.")
    else:
        logger.info("Database connectivity confirmed.")


# --- Routes ------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Basic liveness/info endpoint."""
    return {
        "service": "Anima API",
        "version": app.version,
        "status": "online",
    }


@app.get("/health")
def health(conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Readiness check that verifies a live query against SQL Server."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        cursor.fetchone()
        return {"status": "healthy", "database": DB_NAME}
    except pyodbc.Error as exc:
        logger.exception("Health check DB query failed.")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


# =============================================================================
# Authentication / RBAC / OTP
# =============================================================================

# --- Request / response models ----------------------------------------------
class PatientProfile(BaseModel):
    """Demographic (Patient) + DSM-5 (DSM5_Assessment) data for a patient.

    Used both for an adult Patient registering themselves and for a child
    patient a Guardian creates on registration. All fields optional.
    """
    # Demographics -> Patient
    name: Optional[str] = None
    biological_sex: Optional[int] = None
    age: Optional[float] = None
    # DSM-5 / phenotypic -> DSM5_Assessment
    ground_truth_dx: Optional[int] = None
    adhd_index: Optional[int] = None
    inattentive_score: Optional[int] = None
    hyperactive_score: Optional[int] = None
    iq_measure: Optional[int] = None
    med_status: Optional[int] = None
    nlp_risk_score: Optional[float] = None
    final_combined_score: Optional[float] = None
    # Raw questionnaire answers (JSON: object, array, or pre-serialised string)
    raw_answers: Optional[Union[dict, list, str]] = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)
    # Requested role: only 'Patient' or 'Guardian' honoured (kingston -> Clinician).
    role: Optional[str] = None
    # Display name (Clinician/Guardian require a name; falls back to email local part).
    name: Optional[str] = None
    # Adult patient's own demographic/DSM-5 data.
    profile: Optional[PatientProfile] = None
    # Guardian: link to an existing child patient_ID.
    link_existing_child_id: Optional[str] = None
    # Guardian: create a brand-new child patient with this data.
    new_child: Optional[PatientProfile] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class VerifyOtpRequest(BaseModel):
    user_ID: str
    otp: str


# --- Helpers -----------------------------------------------------------------
def _next_numeric_id(cursor: pyodbc.Cursor, table: str, id_col: str) -> str:
    """Return the next available zero-padded numeric ID for a role table.

    e.g. if MAX(clinician_ID) = '0010005' -> '0010006'. Starts at '0000001'.
    Table/column names are trusted internal constants (never user input).
    """
    cursor.execute(f"SELECT MAX(TRY_CONVERT(BIGINT, {id_col})) FROM dbo.{table};")
    current = cursor.fetchone()[0] or 0
    return str(current + 1).zfill(ROLE_ID_WIDTH)


def _generate_otp() -> str:
    """Cryptographically-random 6-digit one-time passcode as a string."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _issue_otp(cursor: pyodbc.Cursor, user_id: str) -> str:
    """Insert a fresh OTP for the user and return the plaintext code."""
    otp = _generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=OTP_TTL_MINUTES)
    cursor.execute(
        "INSERT INTO dbo.User_OTP (user_ID, otp_code, expires_at) VALUES (?, ?, ?);",
        user_id, otp, expires_at,
    )
    return otp


def _serialise_json(value: Union[dict, list, str, None]) -> Optional[str]:
    """Normalise a raw_answers value into a JSON string for NVARCHAR storage."""
    if value is None:
        return None
    if isinstance(value, str):
        return value  # assumed already JSON; DB ISJSON CHECK validates it
    return json.dumps(value)


def _insert_patient(cursor: pyodbc.Cursor, patient_id: str, user_id: Optional[str],
                    guardian_id: Optional[str], is_child: bool,
                    profile: Optional[PatientProfile]) -> None:
    """Insert a row into the Patient table."""
    p = profile or PatientProfile()
    cursor.execute(
        """
        INSERT INTO dbo.Patient
            (patient_ID, user_ID, guardian_ID, name, biological_sex, is_child, age)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        patient_id, user_id, guardian_id, p.name, p.biological_sex,
        1 if is_child else 0, p.age,
    )


def _insert_dsm5(cursor: pyodbc.Cursor, patient_id: str,
                 profile: Optional[PatientProfile]) -> bool:
    """Insert a DSM5_Assessment row if the profile carries any DSM-5 data.

    Returns True if a row was inserted, False if there was nothing to store.
    """
    if profile is None:
        return False
    dsm5_fields = (
        profile.ground_truth_dx, profile.adhd_index, profile.inattentive_score,
        profile.hyperactive_score, profile.iq_measure, profile.med_status,
        profile.nlp_risk_score, profile.final_combined_score, profile.raw_answers,
    )
    if all(f is None for f in dsm5_fields):
        return False
    cursor.execute(
        """
        INSERT INTO dbo.DSM5_Assessment
            (patient_ID, ground_truth_dx, adhd_index, inattentive_score,
             hyperactive_score, iq_measure, med_status, nlp_risk_score,
             final_combined_score, raw_answers)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        patient_id, profile.ground_truth_dx, profile.adhd_index,
        profile.inattentive_score, profile.hyperactive_score, profile.iq_measure,
        profile.med_status, profile.nlp_risk_score, profile.final_combined_score,
        _serialise_json(profile.raw_answers),
    )
    return True


def _email_exists(cursor: pyodbc.Cursor, email: str) -> bool:
    cursor.execute("SELECT 1 FROM dbo.Users WHERE email = ?;", email)
    return cursor.fetchone() is not None


# --- Router ------------------------------------------------------------------
# Maps to the "FastAPI Router" layer in the system architecture. Auth routes are
# grouped under /api/auth ("api/auth - RBAC Authentication"), alongside the
# future /api/ingest and /api/analysis routers.
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@auth_router.post("/register")
def register(payload: RegisterRequest, conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Register a Patient, Guardian, or (for @kingston.ac.uk emails) a Clinician.

    Admins are never self-registered - they are inserted directly into the DB.
    Returns the new user_ID and a 6-digit OTP for the PowerApps/Outlook client
    to email to the user.
    """
    cursor = conn.cursor()
    email = payload.email.strip().lower()

    # --- Resolve the effective role (RBAC) -----------------------------------
    if email.endswith(CLINICIAN_EMAIL_DOMAIN):
        # Institutional email: force Clinician regardless of requested role.
        role = "Clinician"
    else:
        requested = (payload.role or "").strip().capitalize()
        if requested == "Admin":
            raise HTTPException(
                status_code=403,
                detail="Admin accounts cannot be self-registered.",
            )
        if requested not in SELF_REGISTERABLE_ROLES:
            raise HTTPException(
                status_code=400,
                detail="Requested role must be 'Patient' or 'Guardian'.",
            )
        role = requested

    if _email_exists(cursor, email):
        raise HTTPException(status_code=409, detail="Email is already registered.")

    # Name is required for Clinician/Guardian (NOT NULL); default to email local part.
    display_name = (payload.name or "").strip() or email.split("@", 1)[0]

    try:
        password_hash = hash_password(payload.password)

        # --- Create the role profile + Users supertype row -------------------
        if role == "Clinician":
            role_id = _next_numeric_id(cursor, "Clinician", "clinician_ID")
            user_id = ROLE_PREFIX["Clinician"] + role_id
            _insert_user(cursor, user_id, email, password_hash, "Clinician")
            cursor.execute(
                """
                INSERT INTO dbo.Clinician (clinician_ID, user_ID, name, is_verified)
                VALUES (?, ?, ?, 0);
                """,
                role_id, user_id, display_name,
            )
            child_patient_id = None

        elif role == "Patient":
            # Adult patient registering themselves.
            role_id = _next_numeric_id(cursor, "Patient", "patient_ID")
            user_id = ROLE_PREFIX["Patient"] + role_id
            _insert_user(cursor, user_id, email, password_hash, "Patient")
            _insert_patient(cursor, role_id, user_id, guardian_id=None,
                            is_child=False, profile=payload.profile)
            _insert_dsm5(cursor, role_id, payload.profile)
            child_patient_id = None

        else:  # Guardian
            role_id = _next_numeric_id(cursor, "Guardian", "guardian_ID")
            user_id = ROLE_PREFIX["Guardian"] + role_id
            _insert_user(cursor, user_id, email, password_hash, "Guardian")
            cursor.execute(
                "INSERT INTO dbo.Guardian (guardian_ID, user_ID, name) VALUES (?, ?, ?);",
                role_id, user_id, display_name,
            )
            child_patient_id = _handle_guardian_child(cursor, role_id, payload)

        # --- Issue a verification OTP ----------------------------------------
        otp = _issue_otp(cursor, user_id)
        conn.commit()

    except HTTPException:
        conn.rollback()
        raise
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("Registration DB error.")
        raise HTTPException(status_code=400, detail=f"Registration failed: {exc}")
    except Exception:
        conn.rollback()
        logger.exception("Unexpected registration error.")
        raise HTTPException(status_code=500, detail="Internal registration error.")

    response = {"user_ID": user_id, "role": role, "generated_otp": otp}
    if child_patient_id is not None:
        response["child_patient_id"] = child_patient_id
    return response


def _insert_user(cursor: pyodbc.Cursor, user_id: str, email: str,
                 password_hash: str, role: str) -> None:
    """Insert the authentication supertype (Users) row."""
    cursor.execute(
        """
        INSERT INTO dbo.Users (user_ID, email, password_hash, role)
        VALUES (?, ?, ?, ?);
        """,
        user_id, email, password_hash, role,
    )


def _handle_guardian_child(cursor: pyodbc.Cursor, guardian_id: str,
                           payload: RegisterRequest) -> Optional[str]:
    """Link an existing child, or create a new child patient, for a Guardian.

    Returns the linked/created child patient_ID, or None if no child provided.
    """
    # Create a brand-new child patient (+ optional demographic & DSM-5 data).
    if payload.new_child is not None:
        child_id = _next_numeric_id(cursor, "Patient", "patient_ID")
        _insert_patient(cursor, child_id, user_id=None, guardian_id=guardian_id,
                        is_child=True, profile=payload.new_child)
        _insert_dsm5(cursor, child_id, payload.new_child)
        return child_id

    # Link to an existing child patient.
    if payload.link_existing_child_id:
        child_id = payload.link_existing_child_id.strip()
        cursor.execute(
            "SELECT is_child FROM dbo.Patient WHERE patient_ID = ?;", child_id
        )
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"Child patient '{child_id}' not found.")
        if not row[0]:
            raise HTTPException(status_code=400,
                                detail=f"Patient '{child_id}' is not a child record.")
        cursor.execute(
            "UPDATE dbo.Patient SET guardian_ID = ? WHERE patient_ID = ?;",
            guardian_id, child_id,
        )
        return child_id

    return None


@auth_router.post("/login")
def login(payload: LoginRequest, conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Verify email + password (bcrypt), then issue a fresh login OTP."""
    cursor = conn.cursor()
    email = payload.email.strip().lower()

    cursor.execute(
        "SELECT user_ID, password_hash FROM dbo.Users WHERE email = ?;", email
    )
    row = cursor.fetchone()
    # Verify against the stored bcrypt hash. Uniform error avoids account enumeration.
    if row is None or not verify_password(payload.password, row.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user_id = row.user_ID
    try:
        # Remove any outstanding OTPs, then issue a single new one.
        cursor.execute("DELETE FROM dbo.User_OTP WHERE user_ID = ?;", user_id)
        otp = _issue_otp(cursor, user_id)
        conn.commit()
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("Login OTP issue failed.")
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}")

    return {"user_ID": user_id, "generated_otp": otp}


@auth_router.post("/verify-otp")
def verify_otp(payload: VerifyOtpRequest,
               conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Validate a user's OTP and confirm it has not expired."""
    cursor = conn.cursor()
    user_id = payload.user_ID.strip()
    otp = payload.otp.strip()

    # Match code + user, and require the expiry to still be in the future (UTC).
    cursor.execute(
        """
        SELECT TOP 1 otp_ID
        FROM dbo.User_OTP
        WHERE user_ID = ? AND otp_code = ? AND expires_at >= SYSUTCDATETIME()
        ORDER BY otp_ID DESC;
        """,
        user_id, otp,
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    try:
        # Consume all OTPs for this user so the code cannot be replayed.
        cursor.execute("DELETE FROM dbo.User_OTP WHERE user_ID = ?;", user_id)
        conn.commit()
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("OTP consumption failed.")
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}")

    return {"user_ID": user_id, "verified": True}


# =============================================================================
# Data ingestion  (SysArchitecture: "api/ingest - Data Pipelines")
# =============================================================================
ingest_router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

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


@ingest_router.post("/csv")
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


# =============================================================================
# Patient records - Clinician-only view (decrypts Name)
# SiteMap/ActivityDiagram: only Clinicians may view patient names in the DB.
# =============================================================================
patients_router = APIRouter(prefix="/api/patients", tags=["Patients"])


def _require_clinician(cursor: pyodbc.Cursor, requester_user_id: str) -> None:
    """RBAC guard: allow only users whose role is 'Clinician'."""
    cursor.execute(
        "SELECT role FROM dbo.Users WHERE user_ID = ?;", requester_user_id
    )
    row = cursor.fetchone()
    if row is None or row[0] != "Clinician":
        raise HTTPException(
            status_code=403,
            detail="Only clinicians may view patient names.",
        )


@patients_router.get("")
def list_patients(requester_user_id: str,
                  conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Return patient records with the Name decrypted - Clinicians only.

    ``requester_user_id`` must resolve to a user with role 'Clinician'; the
    Fernet-encrypted Name is decrypted only for that authorised view.
    """
    cursor = conn.cursor()
    _require_clinician(cursor, requester_user_id)

    cursor.execute(
        """
        SELECT patient_ID, name, biological_sex, is_child, age
        FROM dbo.Patient;
        """
    )
    patients = []
    for r in cursor.fetchall():
        patients.append({
            "patient_ID": r.patient_ID,
            # Decrypt for the authorised clinician view.
            "name": decrypt_value(r.name) if r.name else None,
            "biological_sex": r.biological_sex,
            "is_child": r.is_child,
            "age": float(r.age) if r.age is not None else None,
        })

    return {"status": "success", "count": len(patients), "patients": patients}


# --- Wire the routers into the application (main.py = application entry) ------
app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(patients_router)