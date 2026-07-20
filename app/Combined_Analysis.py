"""
Anima - Combined Risk Analysis (orchestration layer).

Fuses the two machine-learning channels into a single ADHD decision for a
clinician: the text & demographic model (DSM5_Analysis) and the image
classification model (MRI_Analysis). It is the layer FastAPI calls when a
clinician triggers "Combined Risk Analysis" on a patient.

How it works
------------
1. Runs each component analysis through its OWN engine - `DSM5_Analysis.
   analyze_patient` and `MRI_Analysis.analyze_mri`. Each opens its own
   connection, scores, and persists its own risk; this module never re-implements
   their logic (clean separation, same as the model/serving split).
2. Fuses the two risks (both 0-100) into `final_combined_score` with configurable
   weights. The text/demographic channel is the stronger signal for ADHD, so it
   is weighted higher by default; the image channel is a modest contributor.
3. Carries the DSM-5 subtype prediction (Inattentive / Hyperactive / Combined)
   through - a single aggregate score can't separate the presentations, so the
   subtype comes from the subscale-based DSM-5 logic.
4. Writes ONE append-only row to `Analysis_Result` - the authoritative, never-
   updated audit trail for the Admin accountability persona (each run is a new
   row; previous runs are kept).

Graceful degradation
---------------------
If the image model has no trained weights yet ("pending") or the patient has no
MRI scans, the combined score falls back to the text/demographic risk alone and
records that in the response + audit row - it never fails just because imaging
isn't ready. (The DSM-5 assessment is required; a patient with none is a 404.)

XAI / explainability is intentionally NOT included here yet.
"""

import os
import logging

from fastapi import APIRouter, HTTPException, Depends

import pyodbc

from database import get_connection, get_db

logger = logging.getLogger("anima.analysis.combined")

router = APIRouter(prefix="/api/analysis", tags=["Analysis - Combined"])

# Fusion weights (text/demographic channel dominant - it is the stronger ADHD
# signal; imaging is a modest contributor). Configurable via env.
NLP_WEIGHT = float(os.getenv("COMBINED_NLP_WEIGHT", "0.7"))
MRI_WEIGHT = float(os.getenv("COMBINED_MRI_WEIGHT", "0.3"))
# Combined risk (0-100) at/above which the aggregate call is "ADHD".
COMBINED_THRESHOLD = float(os.getenv("COMBINED_DIAGNOSIS_THRESHOLD", "50"))
MODEL_VERSION = os.getenv("COMBINED_MODEL_VERSION", "combined_v1")


def _fuse(nlp_risk, mri_risk):
    """Weighted-average the available component risks -> (score, weighting note).

    Uses whichever components are present. With both, a weighted mean (NLP-heavy);
    with only one, that one carries the score. Returns (None, ...) if neither.
    """
    if nlp_risk is not None and mri_risk is not None:
        total = NLP_WEIGHT + MRI_WEIGHT
        score = (nlp_risk * NLP_WEIGHT + mri_risk * MRI_WEIGHT) / total if total else None
        return (round(score, 2) if score is not None else None,
                f"nlp={NLP_WEIGHT:g}, mri={MRI_WEIGHT:g}")
    if nlp_risk is not None:
        return round(float(nlp_risk), 2), "nlp=1.0 (mri unavailable)"
    if mri_risk is not None:
        return round(float(mri_risk), 2), "mri=1.0 (nlp unavailable)"
    return None, "none"


def _run_components(patient_id: str):
    """Run both engines. DSM-5 is required (404 propagates); MRI degrades to None."""
    from DSM5_Analysis import analyze_patient
    from MRI_Analysis import analyze_mri

    # Text & demographic model (required). A missing assessment -> 404.
    nlp_result = analyze_patient(patient_id)
    nlp_risk = nlp_result.get("nlp_risk_score")
    subtype = nlp_result.get("predicted_subtype")

    # Image model (optional). No scans (404) or untrained ("pending") -> None.
    mri_risk, mri_status = None, "unavailable"
    try:
        mri_result = analyze_mri(patient_id)
        mri_status = mri_result.get("status", "unavailable")
        if mri_status == "success":
            mri_risk = mri_result.get("mri_risk_score")
    except HTTPException as exc:
        if exc.status_code == 404:
            mri_status = "no_scans"      # patient has no current MRI - NLP only
        else:
            raise

    return nlp_risk, mri_risk, subtype, mri_status


def analyze_combined(patient_id: str, created_by: str = None) -> dict:
    """Run + fuse both models, persist an Analysis_Result audit row, return result."""
    nlp_risk, mri_risk, subtype, mri_status = _run_components(patient_id)

    combined, weighting = _fuse(nlp_risk, mri_risk)
    if combined is None:
        raise HTTPException(
            status_code=422,
            detail=f"No component scores available for patient '{patient_id}'.",
        )
    diagnosis = "ADHD" if combined >= COMBINED_THRESHOLD else "Control"

    # Persist ONE append-only audit row (with the component IDs for traceability).
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TOP 1 assessment_ID FROM dbo.DSM5_Assessment "
            "WHERE patient_ID = ? ORDER BY assessment_ID DESC;", patient_id)
        row = cursor.fetchone()
        assessment_id = row[0] if row else None

        cursor.execute(
            "SELECT TOP 1 MRI_ID FROM dbo.MRI "
            "WHERE patient_ID = ? AND is_current = 1 ORDER BY MRI_ID DESC;", patient_id)
        row = cursor.fetchone()
        mri_id = row[0] if row else None

        cursor.execute(
            """
            INSERT INTO dbo.Analysis_Result
                (patient_ID, assessment_ID, MRI_ID, nlp_risk_score, mri_risk_score,
                 final_combined_score, predicted_subtype, model_version, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            patient_id, assessment_id, mri_id, nlp_risk, mri_risk,
            combined, subtype, MODEL_VERSION, created_by,
        )
        conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        logger.exception("Combined analysis persistence failed.")
        raise HTTPException(status_code=500, detail=f"Combined analysis failed: {exc}")
    finally:
        conn.close()

    return {
        "status": "success",
        "patient_ID": patient_id,
        "nlp_risk_score": nlp_risk,
        "mri_risk_score": mri_risk,
        "mri_status": mri_status,
        "final_combined_score": combined,
        "predicted_diagnosis": diagnosis,
        "predicted_subtype": subtype,
        "weighting": weighting,
        "model_version": MODEL_VERSION,
    }


@router.post("/combined/{patient_id}")
def analyze_combined_endpoint(patient_id: str, requester_user_id: str,
                              conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Run the Combined Risk Analysis for ONE patient and store an audit row.

    Clinicians analyse a single patient at a time - there is no batch trigger.
    RBAC: the requester must be authorised for this patient (Admin, the assigned
    clinician, the linked guardian, or the patient themselves). The requester's
    user_ID is recorded as ``Analysis_Result.created_by`` so the audit trail shows
    who ran each analysis.
    """
    requester = requester_user_id.strip()
    _authorize_patient_access(conn.cursor(), requester, patient_id.strip())
    return analyze_combined(patient_id.strip(), created_by=requester)


# =============================================================================
# Retrieval (read side) - RBAC-gated views of the Analysis_Result audit trail
# =============================================================================
# Analysis_Result columns, in the order the SELECTs below return them.
_RESULT_COLS = (
    "analysis_ID", "patient_ID", "assessment_ID", "MRI_ID", "nlp_risk_score",
    "mri_risk_score", "final_combined_score", "predicted_subtype",
    "model_version", "created_at", "created_by",
)


def _authorize_patient_access(cursor, requester_user_id: str, patient_id: str) -> None:
    """RBAC gate for reading a patient's analysis results. Raises on denial.

    Access rules (also protects child records - a child has no own login, so the
    only paths to it are Admin, its assigned Clinician, or its linked Guardian):
      * Admin      -> full access (the accountability / audit persona);
      * Clinician  -> only patients assigned via Clinician_Patient_Assignment;
      * Guardian   -> only their linked child (Patient.guardian_ID);
      * Patient    -> only themselves (Patient.user_ID).
    """
    cursor.execute("SELECT role FROM dbo.Users WHERE user_ID = ?;", requester_user_id)
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Unknown requester.")
    role = row[0]

    if role == "Admin":
        return   # audit persona: full read access

    # Everyone else needs the patient to exist; fetch its ownership links.
    cursor.execute(
        "SELECT user_ID, guardian_ID FROM dbo.Patient WHERE patient_ID = ?;", patient_id)
    prow = cursor.fetchone()
    if prow is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    p_user_id, p_guardian_id = prow[0], prow[1]

    if role == "Clinician":
        # user_ID is 'C' + clinician_ID (role-prefix convention).
        clinician_id = requester_user_id[1:]
        cursor.execute(
            "SELECT 1 FROM dbo.Clinician_Patient_Assignment "
            "WHERE clinician_ID = ? AND patient_ID = ?;", clinician_id, patient_id)
        if cursor.fetchone() is not None:
            return
    elif role == "Guardian":
        guardian_id = requester_user_id[1:]           # 'G' + guardian_ID
        if p_guardian_id is not None and str(p_guardian_id) == guardian_id:
            return
    elif role == "Patient":
        if p_user_id is not None and str(p_user_id) == requester_user_id:
            return

    raise HTTPException(
        status_code=403,
        detail="You are not authorised to view this patient's analysis results.")


def _result_to_dict(row) -> dict:
    """Map an Analysis_Result row (in _RESULT_COLS order) to JSON-safe types."""
    out = {}
    for col, val in zip(_RESULT_COLS, row):
        if col.endswith("_score") and val is not None:
            out[col] = float(val)                     # Decimal -> float
        elif col == "created_at" and val is not None:
            out[col] = val.isoformat()                # datetime -> ISO string
        else:
            out[col] = val
    return out


@router.get("/results/{patient_id}")
def get_latest_result(patient_id: str, requester_user_id: str,
                      conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Latest combined analysis result for a patient (RBAC-gated)."""
    cursor = conn.cursor()
    _authorize_patient_access(cursor, requester_user_id.strip(), patient_id.strip())
    cursor.execute(
        f"SELECT TOP 1 {', '.join(_RESULT_COLS)} FROM dbo.Analysis_Result "
        "WHERE patient_ID = ? ORDER BY analysis_ID DESC;", patient_id.strip())
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis results found for patient '{patient_id}'.")
    return {"status": "success", "patient_ID": patient_id.strip(),
            "result": _result_to_dict(row)}


@router.get("/results/{patient_id}/history")
def get_result_history(patient_id: str, requester_user_id: str,
                       limit: int = 50,
                       conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Full analysis history for a patient, newest first (RBAC-gated audit trail).

    ``limit`` is bounded to [1, 500] here rather than via a Query() default, so the
    function is safe both as an HTTP endpoint and when called directly (the read
    smoke test / other code) - a Query() default only resolves under FastAPI.
    """
    limit = max(1, min(int(limit), 500))
    cursor = conn.cursor()
    _authorize_patient_access(cursor, requester_user_id.strip(), patient_id.strip())
    cursor.execute(
        f"SELECT TOP (?) {', '.join(_RESULT_COLS)} FROM dbo.Analysis_Result "
        "WHERE patient_ID = ? ORDER BY analysis_ID DESC;", limit, patient_id.strip())
    rows = cursor.fetchall()
    return {"status": "success", "patient_ID": patient_id.strip(),
            "count": len(rows), "history": [_result_to_dict(r) for r in rows]}
