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

Explainability (XAI)
--------------------
Every result carries an "explanation" block so the frontend can show WHY the
model decided as it did. Because the trained head is a linear model on top of the
frozen features, the contributions are EXACT and cheap to compute:
  * feature_importance - how much each score/demographic (and the note as a whole)
    pushed the prediction, as impact percentages for a Power Apps bar chart;
  * influential_words - the note's most impactful words with a signed push score.
"""

import os
import logging

from fastapi import APIRouter, HTTPException

from database import get_connection

logger = logging.getLogger("anima.analysis.dsm5")

router = APIRouter(prefix="/api/analysis", tags=["Analysis - DSM5"])

# Path to the trained head artifact.
_HERE = os.path.dirname(os.path.abspath(__file__))
HEAD_PATH = os.getenv("DSM5_HEAD_PATH", os.path.join(_HERE, "models", "dsm5_head.pt"))

# Conners' T-score at/above which a subscale is treated as clinically elevated.
SUBTYPE_THRESHOLD = float(os.getenv("DSM5_SUBTYPE_THRESHOLD", "60"))
# T-score range used to map severity onto a 0-100 risk scale.
T_MIN, T_MAX = 40.0, 90.0
# How many top influential words to return from the clinical note.
TOP_WORDS = int(os.getenv("DSM5_TOP_WORDS", "8"))
# Human-friendly labels for the structured features (no raw column names in the UI).
FEATURE_LABELS = {
    "age": "Age",
    "biological_sex": "Biological sex",
    "is_child": "Child",
    "inattentive_score": "Inattentive score",
    "hyperactive_score": "Hyperactive score",
}


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


def _score_and_explain(age, biological_sex, is_child,
                       hyperactive_score, inattentive_score, clinical_notes):
    """Score with the trained head AND build its explanation; or (None, None).

    Loads the head via ``dsm5_model.load_head`` (the correct {config, state_dict}
    format) so both the score and the exact feature/word contributions come from
    the real trained model. Lazily imports torch/features so a failure degrades
    gracefully to the heuristic instead of taking the endpoint down.
    """
    if not os.path.exists(HEAD_PATH):
        return None, None
    try:
        import torch
        import dsm5_features
        import dsm5_model

        features = dsm5_features.build_feature_vector(
            age, biological_sex, is_child,
            hyperactive_score, inattentive_score, clinical_notes,
        )
        head = dsm5_model.load_head(HEAD_PATH)              # rebuilds DSM5Head + eval()
        with torch.no_grad():
            logit = head(features.unsqueeze(0))
        risk = round(float(torch.sigmoid(logit).item()) * 100.0, 2)
        explanation = _explain_text(head, features, clinical_notes)
        return risk, explanation
    except Exception:
        logger.exception("Trained-head scoring/explanation failed; using heuristic.")
        return None, None


def _explain_text(head, features, clinical_notes) -> dict:
    """Exact per-feature + per-word contributions to the ADHD logit (linear head).

    feature_importance: each structured feature's contribution ``w_i*(x_i-mean_i)``
    plus the note block's total, ranked and turned into impact percentages.
    influential_words: each word's push = its embedding dotted with the model's
    note weight (the note embedding is a mean of these word vectors).
    """
    import torch
    import dsm5_features

    with torch.no_grad():
        proj = head.project(features.unsqueeze(0))                    # (1, feat_dim)
        z = (proj - head.feat_mean) / head.feat_std
        contrib = head.linear.weight.detach().squeeze(0) * z.squeeze(0)   # (feat_dim,)
    sd = head.structured_dim

    items = [(FEATURE_LABELS.get(name, name), float(contrib[i]))
             for i, name in enumerate(dsm5_features.STRUCTURED_FEATURE_NAMES)]
    items.append(("Clinical notes", float(contrib[sd:].sum())))       # note as one bar
    total = sum(abs(v) for _, v in items) or 1.0
    feature_importance = [
        {"feature": name,
         "impact_percent": round(abs(v) / total * 100.0, 1),
         "direction": "toward ADHD" if v >= 0 else "toward control"}
        for name, v in sorted(items, key=lambda t: -abs(t[1]))
    ]

    influential_words = []
    words, word_vectors = dsm5_features.embed_note_tokens(clinical_notes)
    if len(words):
        w_eff = head.note_effective_weight()                          # (768,)
        pushes = (word_vectors @ w_eff).tolist()
        strongest = max((abs(p) for p in pushes), default=0.0) or 1.0
        ranked = sorted(zip(words, pushes), key=lambda t: -abs(t[1]))[:TOP_WORDS]
        influential_words = [
            {"word": w,
             "push": round(p / strongest * 100.0, 1),                 # -100..100 relative
             "direction": "toward ADHD" if p >= 0 else "toward control"}
            for w, p in ranked if w.strip()
        ]

    return {
        "method": "model",
        "summary": _text_summary(feature_importance, influential_words),
        "feature_importance": feature_importance,
        "influential_words": influential_words,
    }


def _heuristic_explanation(inattentive_score, hyperactive_score) -> dict:
    """Simple explanation when the trained model isn't available (scores only)."""
    scores = [("Inattentive score", float(inattentive_score or 0)),
              ("Hyperactive score", float(hyperactive_score or 0))]
    total = sum(v for _, v in scores) or 1.0
    feature_importance = [
        {"feature": n, "impact_percent": round(v / total * 100.0, 1),
         "direction": "toward ADHD"}
        for n, v in sorted(scores, key=lambda t: -t[1])
    ]
    return {
        "method": "heuristic",
        "summary": ("Estimated from the questionnaire subscale scores "
                    "(trained text model not loaded)."),
        "feature_importance": feature_importance,
        "influential_words": [],
    }


def _text_summary(feature_importance, influential_words) -> str:
    """One plain-language sentence naming the top drivers."""
    if not feature_importance:
        return "No explanation available."
    top = feature_importance[0]
    msg = (f"The biggest factor was '{top['feature']}' "
           f"({top['impact_percent']}% {top['direction']}).")
    toward = [w["word"] for w in influential_words if w["direction"] == "toward ADHD"][:3]
    if toward:
        msg += " Notable words in the note: " + ", ".join(toward) + "."
    return msg


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

        # Score + explain: trained model if available, else the transparent heuristic.
        risk, explanation = _score_and_explain(
            age, biological_sex, is_child,
            hyperactive_score, inattentive_score, clinical_notes)
        scoring_method = "model"
        if risk is None:
            risk = _heuristic_score(inattentive_score, hyperactive_score)
            explanation = _heuristic_explanation(inattentive_score, hyperactive_score)
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
            # XAI: why the text model decided this (Power Apps binds to these).
            "explanation": {"text_model": explanation},
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