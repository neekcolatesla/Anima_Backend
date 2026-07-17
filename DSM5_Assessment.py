"""
Anima - DSM-5 assessment intake (questionnaire + clinical notes).

Owns the /api/dsm5 router. Two per-patient input paths write into the
DSM5_Assessment table's raw_answers (questionnaire JSON) and clinician_notes
(free-text NLP narrative):

  * POST /api/dsm5/questionnaire     - patient (self) or guardian (their child)
                                       submits the 18 DSM-5 answers + narrative.
  * PUT  /api/dsm5/clinician-notes   - a Clinician edits/adds their own clinical
                                       observations for ANY patient.

Bulk CSV ingestion of DSM-5 data lives in CSV_Ingestion.py (/api/ingest/dsm5-csv)
and reuses the shared helpers exported here.
"""

import json
import logging
from typing import Optional, List, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

import pyodbc

from database import get_db
from Auth_RBAC import _require_clinician

logger = logging.getLogger("anima.dsm5")

router = APIRouter(prefix="/api/dsm5", tags=["DSM-5 Assessment"])


# =============================================================================
# The 18 standard DSM-5 ADHD items (child + adult wording).
# Canonical in-app copy; the standalone generator script keeps its own.
# =============================================================================
DSM5_QUESTIONS = {
    "inattentive": [
        {"id": "IA1",
         "child": "Fails to give close attention to details or makes careless mistakes in schoolwork.",
         "adult": "Overlooks details or makes careless mistakes at work or in other activities."},
        {"id": "IA2",
         "child": "Has difficulty keeping attention on tasks, schoolwork, or play.",
         "adult": "Has difficulty sustaining attention in tasks, meetings, or lengthy reading."},
        {"id": "IA3",
         "child": "Does not seem to listen when spoken to directly.",
         "adult": "Does not seem to listen even when spoken to directly."},
        {"id": "IA4",
         "child": "Does not follow through on instructions and fails to finish schoolwork or chores.",
         "adult": "Fails to follow through on instructions and does not finish work duties."},
        {"id": "IA5",
         "child": "Has difficulty organising tasks, belongings, and activities.",
         "adult": "Has difficulty organising tasks, time, and workspace."},
        {"id": "IA6",
         "child": "Avoids or dislikes tasks that require sustained mental effort, like homework.",
         "adult": "Avoids or dislikes tasks needing sustained mental effort, like reports or forms."},
        {"id": "IA7",
         "child": "Loses things needed for tasks (e.g. pencils, books, toys).",
         "adult": "Loses things needed for tasks (e.g. keys, phone, paperwork, wallet)."},
        {"id": "IA8",
         "child": "Is easily distracted by noises or things going on around them.",
         "adult": "Is easily distracted by external stimuli or unrelated thoughts."},
        {"id": "IA9",
         "child": "Is forgetful in daily activities and routines.",
         "adult": "Is forgetful in daily activities (appointments, chores, returning calls)."},
    ],
    "hyperactive": [
        {"id": "HI1",
         "child": "Fidgets with or taps hands or feet, or squirms in their seat.",
         "adult": "Fidgets with or taps hands or feet, or squirms when seated."},
        {"id": "HI2",
         "child": "Leaves their seat when staying seated is expected (e.g. in class).",
         "adult": "Leaves their seat in situations where staying seated is expected."},
        {"id": "HI3",
         "child": "Runs about or climbs in situations where it is inappropriate.",
         "adult": "Feels restless or has difficulty sitting still for long periods."},
        {"id": "HI4",
         "child": "Is unable to play or take part in activities quietly.",
         "adult": "Is unable to engage in leisure activities quietly."},
        {"id": "HI5",
         "child": "Is often 'on the go' or acts as if driven by a motor.",
         "adult": "Feels driven to keep moving, as if propelled by a motor."},
        {"id": "HI6",
         "child": "Talks excessively.",
         "adult": "Talks excessively."},
        {"id": "HI7",
         "child": "Blurts out answers before questions have been finished.",
         "adult": "Blurts out answers or completes others' sentences before they finish."},
        {"id": "HI8",
         "child": "Has difficulty waiting for their turn.",
         "adult": "Has difficulty waiting their turn (e.g. in queues)."},
        {"id": "HI9",
         "child": "Interrupts or intrudes on others' games or conversations.",
         "adult": "Interrupts or intrudes on others (conversations, activities)."},
    ],
}


# =============================================================================
# Shared helpers (reused by CSV_Ingestion.py bulk loader)
# =============================================================================
def patient_exists(cursor: pyodbc.Cursor, patient_id: str) -> bool:
    cursor.execute("SELECT 1 FROM dbo.Patient WHERE patient_ID = ?;", patient_id)
    return cursor.fetchone() is not None


def latest_assessment_id(cursor: pyodbc.Cursor, patient_id: str) -> Optional[int]:
    """Return the most recent DSM5_Assessment.assessment_ID for a patient, or None."""
    cursor.execute(
        """
        SELECT TOP 1 assessment_ID FROM dbo.DSM5_Assessment
        WHERE patient_ID = ? ORDER BY assessment_ID DESC;
        """,
        patient_id,
    )
    row = cursor.fetchone()
    return row[0] if row else None


def build_raw_answers(is_child: bool, ia_answers, hi_answers) -> dict:
    """Assemble the raw_answers JSON (same shape as the synthetic generator)."""
    key = "child" if is_child else "adult"

    def block(subscale, answers):
        return {
            q["id"]: {"question": q[key], "answer": int(a)}
            for q, a in zip(DSM5_QUESTIONS[subscale], answers)
        }

    return {
        "instrument": "DSM-5 ADHD symptom checklist (18-item)",
        "form": key,
        "likert_scale": {"0": "Never", "1": "Rarely", "2": "Sometimes",
                         "3": "Often", "4": "Very often"},
        "inattentive": block("inattentive", ia_answers),
        "hyperactive": block("hyperactive", hi_answers),
        "inattentive_sum": int(sum(ia_answers)),
        "hyperactive_sum": int(sum(hi_answers)),
        "total_sum": int(sum(ia_answers) + sum(hi_answers)),
        "source": "powerapps_questionnaire",
    }


def upsert_narrative(cursor: pyodbc.Cursor, patient_id: str,
                     raw_answers_json: Optional[str] = None,
                     clinician_notes: Optional[str] = None,
                     set_raw: bool = False, set_notes: bool = False) -> str:
    """Update the patient's latest DSM5_Assessment (or insert one) with the given
    raw_answers and/or clinician_notes. Returns 'updated' or 'inserted'.

    ``set_raw`` / ``set_notes`` control which columns are written so an update
    never clobbers a field the caller didn't intend to change.
    """
    aid = latest_assessment_id(cursor, patient_id)
    if aid is not None:
        sets, params = [], []
        if set_raw:
            sets.append("raw_answers = ?"); params.append(raw_answers_json)
        if set_notes:
            sets.append("clinician_notes = ?"); params.append(clinician_notes)
        if sets:
            params.append(aid)
            cursor.execute(
                f"UPDATE dbo.DSM5_Assessment SET {', '.join(sets)} WHERE assessment_ID = ?;",
                *params,
            )
        return "updated"

    cursor.execute(
        "INSERT INTO dbo.DSM5_Assessment (patient_ID, raw_answers, clinician_notes) VALUES (?, ?, ?);",
        patient_id,
        raw_answers_json if set_raw else None,
        clinician_notes if set_notes else None,
    )
    return "inserted"


def _require_self_or_guardian(cursor: pyodbc.Cursor, requester_user_id: str,
                              patient_id: str) -> str:
    """RBAC for questionnaire submission.

    A Patient may submit only for their own record; a Guardian only for their
    linked child (is_child = 1). Returns the requester's role, else raises 403.
    """
    cursor.execute("SELECT role FROM dbo.Users WHERE user_ID = ?;", requester_user_id)
    r = cursor.fetchone()
    if r is None:
        raise HTTPException(status_code=403, detail="Unknown requester.")
    role = r[0]

    if role == "Patient":
        cursor.execute(
            "SELECT patient_ID FROM dbo.Patient WHERE user_ID = ?;", requester_user_id
        )
        pr = cursor.fetchone()
        if pr is None or pr[0] != patient_id:
            raise HTTPException(
                status_code=403,
                detail="Patients may only submit their own questionnaire.",
            )
    elif role == "Guardian":
        cursor.execute(
            "SELECT guardian_ID FROM dbo.Guardian WHERE user_ID = ?;", requester_user_id
        )
        gr = cursor.fetchone()
        if gr is None:
            raise HTTPException(status_code=403, detail="Guardian profile not found.")
        cursor.execute(
            """
            SELECT 1 FROM dbo.Patient
            WHERE patient_ID = ? AND guardian_ID = ? AND is_child = 1;
            """,
            patient_id, gr[0],
        )
        if cursor.fetchone() is None:
            raise HTTPException(
                status_code=403,
                detail="Guardians may only submit for their linked child.",
            )
    else:
        raise HTTPException(
            status_code=403,
            detail="Only patients or guardians may submit the questionnaire.",
        )
    return role


# =============================================================================
# Request models
# =============================================================================
class QuestionnaireRequest(BaseModel):
    requester_user_id: str
    patient_id: str
    # 9 inattentive + 9 hyperactive Likert answers (0-4).
    inattentive_answers: List[int]
    hyperactive_answers: List[int]
    # Optional free-text: adult self-report, or guardian/teacher observations.
    notes: Optional[str] = None

    @field_validator("inattentive_answers", "hyperactive_answers")
    @classmethod
    def _check_answers(cls, v):
        if len(v) != 9:
            raise ValueError("must contain exactly 9 answers")
        if any(a < 0 or a > 4 for a in v):
            raise ValueError("each answer must be an integer 0-4")
        return v


class ClinicianNotesRequest(BaseModel):
    requester_user_id: str
    patient_id: str
    notes: str = Field(min_length=1)
    # 'append' (default) preserves prior narrative; 'replace' overwrites it.
    mode: Literal["append", "replace"] = "append"


# =============================================================================
# Endpoints
# =============================================================================
@router.post("/questionnaire")
def submit_questionnaire(payload: QuestionnaireRequest,
                         conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Patient/guardian submits the 18 DSM-5 answers (+ optional narrative).

    The narrative (adult self-report or guardian/teacher observation) is stored
    in clinician_notes; the answers are stored as raw_answers JSON.
    """
    cursor = conn.cursor()
    patient_id = payload.patient_id.strip()

    # RBAC: patient-for-self or guardian-for-their-child.
    _require_self_or_guardian(cursor, payload.requester_user_id.strip(), patient_id)

    # Pick child/adult wording from the patient record.
    cursor.execute("SELECT is_child FROM dbo.Patient WHERE patient_ID = ?;", patient_id)
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    is_child = bool(row[0])

    raw_answers = build_raw_answers(is_child, payload.inattentive_answers,
                                    payload.hyperactive_answers)
    raw_json = json.dumps(raw_answers, ensure_ascii=False)
    notes = payload.notes.strip() if payload.notes else None

    try:
        outcome = upsert_narrative(
            cursor, patient_id,
            raw_answers_json=raw_json, clinician_notes=notes,
            set_raw=True, set_notes=notes is not None,
        )
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("Questionnaire submission DB error.")
        raise HTTPException(status_code=400, detail=f"Submission failed: {exc}")

    return {
        "status": "success",
        "message": f"Questionnaire {outcome} for patient {patient_id}.",
        "patient_ID": patient_id,
        "form": raw_answers["form"],
        "inattentive_sum": raw_answers["inattentive_sum"],
        "hyperactive_sum": raw_answers["hyperactive_sum"],
    }


@router.put("/clinician-notes")
def edit_clinician_notes(payload: ClinicianNotesRequest,
                         conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Clinician edits/adds their own clinical observations for ANY patient.

    'append' (default) attributes and appends to the existing narrative;
    'replace' overwrites it with the attributed note.
    """
    cursor = conn.cursor()
    requester = payload.requester_user_id.strip()
    patient_id = payload.patient_id.strip()

    # RBAC: clinicians only (any patient).
    _require_clinician(cursor, requester)

    if not patient_exists(cursor, patient_id):
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    attributed = f"[Clinician {requester}]: {payload.notes.strip()}"

    try:
        aid = latest_assessment_id(cursor, patient_id)
        if aid is None:
            cursor.execute(
                "INSERT INTO dbo.DSM5_Assessment (patient_ID, clinician_notes) VALUES (?, ?);",
                patient_id, attributed,
            )
            new_notes = attributed
        else:
            if payload.mode == "replace":
                new_notes = attributed
            else:  # append to any existing narrative
                cursor.execute(
                    "SELECT clinician_notes FROM dbo.DSM5_Assessment WHERE assessment_ID = ?;",
                    aid,
                )
                existing = cursor.fetchone()[0]
                new_notes = (existing + "\n\n" + attributed) if existing else attributed
            cursor.execute(
                "UPDATE dbo.DSM5_Assessment SET clinician_notes = ? WHERE assessment_ID = ?;",
                new_notes, aid,
            )
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("Clinician notes DB error.")
        raise HTTPException(status_code=400, detail=f"Update failed: {exc}")

    return {
        "status": "success",
        "message": f"Clinician notes {payload.mode}d for patient {patient_id}.",
        "patient_ID": patient_id,
        "clinician_notes": new_notes,
    }