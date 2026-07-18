"""
Anima - DSM-5 / text & demographic analysis engine (inference).

Owns the /api/analysis/dsm5 endpoint. For a given patient it pulls the latest
demographic + DSM-5 assessment record, turns it into features via the shared
dsm5_features module, scores an NLP ADHD-risk value, guesses the ADHD subtype
from the subscale scores, writes the risk back to DSM5_Assessment, and returns
the result.

Robustness by design
--------------------
* The heavy path (Bio_ClinicalBERT + the trained head) is imported LAZILY and
  wrapped in try/except, so the API boots and still scores even if torch /
  transformers / HuggingFace are unavailable, or no trained head exists yet.
* When the trained head is missing or fails, it falls back to a transparent
  heuristic computed purely from the Conners' subscale T-scores - no model
  needed - so this endpoint never crashes on a scoring failure.

(SysArchitecture: "Machine Learning Engine -> Text & Demographic Model".)
XAI / explainability is intentionally NOT included here yet.
"""

import os
import logging

from fastapi import APIRouter, HTTPException

from database import get_connection

logger = logging.getLogger("anima.analysis.dsm5")

router = APIRouter(prefix="/api/analysis", tags=["Analysis - DSM5"])

# Path to the trained head artifact (produced later on the training branch).
_HERE = os.path.dirname(os.path.abspath(__file__))
HEAD_PATH = os.getenv("DSM5_HEAD_PATH", os.path.join(_HERE, "models", "dsm5_head.pt"))

# Conners' T-score at/above which a subscale is treated as clinically elevated.
SUBTYPE_THRESHOLD = float(os.getenv("DSM5_SUBTYPE_THRESHOLD", "60"))
# T-score range used to map severity onto a 0-100 risk scale.
T_MIN, T_MAX = 40.0, 90.0


# =============================================================================
# Scoring helpers
# =============================================================================
def _heuristic_score(inattentive_score, hyperactive_score) -> float:
    """Fallback NLP risk (0-100) from the subscale T-scores alone.

    Uses the higher of the two subscales (peak severity) mapped from the clinical
    T-score range (40-90) onto 0-100. Requires no model, so it always works.
    """
    scores = [s for s in (inattentive_score, hyperactive_score) if s is not None]
    if not scores:
        return 0.0
    peak = max(float(s) for s in scores)
    risk = (peak - T_MIN) / (T_MAX - T_MIN) * 100.0
    return round(min(max(risk, 0.0), 100.0), 2)


def _score_with_model(age, biological_sex, is_child,
                      hyperactive_score, inattentive_score, clinical_notes):
    """Score via the trained hybrid head, or return None to trigger the fallback.

    Lazily imports torch + the shared feature module so an import/download/model
    failure degrades gracefully instead of taking down the endpoint. The head
    contract (for the training branch): a torch module mapping a (1, 773) feature
    tensor to a single logit; risk = sigmoid(logit) * 100.
    """
    if not os.path.exists(HEAD_PATH):
        return None
    try:
        import torch
        import dsm5_features

        features = dsm5_features.build_feature_vector(
            age, biological_sex, is_child,
            hyperactive_score, inattentive_score, clinical_notes,
        )
        head = torch.load(HEAD_PATH, map_location="cpu")
        head.eval()
        with torch.no_grad():
            logit = head(features.unsqueeze(0))
        prob = torch.sigmoid(logit).item()
        return round(float(prob) * 100.0, 2)
    except Exception:
        logger.exception("Trained-head scoring failed; using heuristic fallback.")
        return None


def _predict_subtype(inattentive_score, hyperactive_score) -> str:
    """Basic if/else rules over the two subscale scores -> ADHD subtype.

    A single aggregate risk can't separate the presentations, so the subtype is
    decided by which subscale(s) are clinically elevated (>= SUBTYPE_THRESHOLD).
    """
    inattentive_high = (inattentive_score or 0) >= SUBTYPE_THRESHOLD
    hyperactive_high = (hyperactive_score or 0) >= SUBTYPE_THRESHOLD

    if inattentive_high and hyperactive_high:
        return "Combined"
    if inattentive_high:
        return "Inattentive"
    if hyperactive_high:
        return "Hyperactive"
    return "None"   # neither subscale elevated -> no ADHD subtype indicated


# =============================================================================
# Core: connect, fetch, score, persist, return
# =============================================================================
def analyze_patient(patient_id: str) -> dict:
    """Fetch a patient's latest DSM-5 record, score NLP risk + subtype, persist.

    Opens its own DB connection and closes it safely. Returns the risk score,
    the predicted subtype, and which scoring path was used.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Pull demographics (Patient) + the latest assessment (DSM5_Assessment).
        cursor.execute(
            """
            SELECT TOP 1
                p.age, p.biological_sex, p.is_child,
                d.assessment_ID, d.inattentive_score, d.hyperactive_score,
                d.clinician_notes
            FROM dbo.Patient AS p
            JOIN dbo.DSM5_Assessment AS d ON d.patient_ID = p.patient_ID
            WHERE p.patient_ID = ?
            ORDER BY d.assessment_ID DESC;
            """,
            patient_id,
        )
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No DSM-5 assessment found for patient '{patient_id}'.",
            )

        (age, biological_sex, is_child, assessment_id,
         inattentive_score, hyperactive_score, clinical_notes) = row

        # Score: trained model if available, else the transparent heuristic.
        risk = _score_with_model(age, biological_sex, is_child,
                                 hyperactive_score, inattentive_score, clinical_notes)
        scoring_method = "model"
        if risk is None:
            risk = _heuristic_score(inattentive_score, hyperactive_score)
            scoring_method = "heuristic"

        subtype = _predict_subtype(inattentive_score, hyperactive_score)

        # Persist the NLP risk score back onto the assessment row.
        cursor.execute(
            "UPDATE dbo.DSM5_Assessment SET nlp_risk_score = ? WHERE assessment_ID = ?;",
            risk, assessment_id,
        )
        conn.commit()

        return {
            "status": "success",
            "patient_ID": patient_id,
            "nlp_risk_score": risk,
            "predicted_subtype": subtype,
            "scoring_method": scoring_method,
        }

    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        logger.exception("DSM-5 analysis failed.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        conn.close()   # always close the connection safely


@router.post("/dsm5/{patient_id}")
def analyze_dsm5(patient_id: str) -> dict:
    """Run the text & demographic model for one patient and store the NLP risk."""
    return analyze_patient(patient_id.strip())