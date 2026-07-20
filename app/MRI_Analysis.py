"""
Anima - MRI image analysis engine (inference).

Owns the /api/analysis/mri endpoint. For a given patient it reads the CURRENT
processed 2D slice stacks (anat + anat_gm) recorded in the MRI table, runs them
through the trained image classifier, aggregates the per-slice outputs into one
patient-level ADHD risk, writes that risk back onto the patient's current MRI
rows, and returns the predicted diagnosis plus a confidence score.

Separation of concerns (important)
----------------------------------
The CNN architecture lives ENTIRELY in ``mri_model.py``. This module never
defines a network - it only loads the saved weights via ``mri_model.load_cnn``
and calls the model. So the classifier and this prediction/serving code stay
completely independent: the training branch can evolve the architecture in
mri_model.py without touching this file, and vice-versa. This mirrors the
DSM-5 split (dsm5_model.py = architecture, DSM5_Analysis.py = inference).

Robustness by design
--------------------
* torch / PIL and the model module are imported LAZILY inside the scoring path,
  so the API boots even if those heavy deps or a trained model are unavailable.
* Until the training branch produces ``models/mri_cnn.pt``, there is no honest
  image-based score to give (unlike the DSM-5 engine, pixels have no transparent
  hand-rule fallback). In that case the endpoint returns a clear "pending"
  status instead of fabricating a number - it never crashes.

(SysArchitecture: "Machine Learning Engine -> Image Classification Model".)
XAI / explainability is intentionally NOT included here yet.
"""

import os
import logging

from fastapi import APIRouter, HTTPException

from database import get_connection
import mri_slices

logger = logging.getLogger("anima.analysis.mri")

router = APIRouter(prefix="/api/analysis", tags=["Analysis - MRI"])

# Path to the trained CNN artifact (produced later on the training branch).
_HERE = os.path.dirname(os.path.abspath(__file__))
CNN_PATH = os.getenv("MRI_CNN_PATH", os.path.join(_HERE, "models", "mri_cnn.pt"))

# Model version tag recorded on any persisted score (for audit / reproducibility).
MODEL_VERSION = os.getenv("MRI_MODEL_VERSION", "mri_cnn_v1")

# Patient-level ADHD risk (0-100) at/above which the image model calls "ADHD".
MRI_DIAGNOSIS_THRESHOLD = float(os.getenv("MRI_DIAGNOSIS_THRESHOLD", "50"))

# Slice selection - MUST match train_mri.py so train and serve score the SAME
# slices (no skew). Defaults mirror the trainer's defaults.
MRI_CENTRAL_FRAC = float(os.getenv("MRI_CENTRAL_FRAC", "0.6"))
MRI_MIN_FOREGROUND = float(os.getenv("MRI_MIN_FOREGROUND", "0.0"))
# Cap on how many paired slices to score per patient (keeps serve-time inference
# bounded; 0 = use every selected slice). Slices are evenly sampled.
MRI_MAX_SLICES = int(os.getenv("MRI_MAX_SLICES", "0"))


# =============================================================================
# Slice loading / batching helpers (pure image handling - no model here)
# =============================================================================
def _build_batch(anat_dir: str, anat_gm_dir: str, input_size: int):
    """Pair anat & anat_gm slices by index into a (N, 2, H, W) float tensor.

    Channel 0 = anat (structural T1), channel 1 = anat_gm (grey matter). Slice
    selection + positional pairing is delegated to the shared mri_slices module,
    the SAME selector the trainer uses (central crop -> optional foreground filter
    -> optional even subsample) so serving scores the same slices training saw.
    Resized to ``input_size`` and scaled to [0, 1]. Returns None if empty.
    Imports torch/PIL lazily so import failures degrade gracefully.
    """
    import numpy as np
    from PIL import Image
    import torch

    pairs = mri_slices.pair_slices(
        anat_dir, anat_gm_dir,
        central_frac=MRI_CENTRAL_FRAC, min_foreground=MRI_MIN_FOREGROUND,
        max_slices=MRI_MAX_SLICES,
    )
    if not pairs:
        return None

    def _load(path):
        img = Image.open(path).convert("L").resize((input_size, input_size))
        return np.asarray(img, dtype=np.float32) / 255.0

    chans = [np.stack([_load(a), _load(g)], axis=0) for a, g in pairs]  # each (2,H,W)
    return torch.from_numpy(np.stack(chans, axis=0))                    # (N, 2, H, W)


# =============================================================================
# Scoring (loads the SEPARATE CNN module - no architecture defined here)
# =============================================================================
def _score_with_cnn(anat_dir: str, anat_gm_dir: str):
    """Run the trained CNN over a patient's paired slices -> (risk, confidence).

    Returns a dict {mri_risk_score, confidence, slices_scored} on success, or
    None to signal the caller to report a "pending / untrained" result (missing
    weights, missing deps, or unreadable slices). risk & confidence are 0-100.
    """
    if not os.path.exists(CNN_PATH):
        return None
    try:
        import torch
        import mri_model   # the SEPARATE architecture module

        model = mri_model.load_cnn(CNN_PATH, map_location="cpu")   # eval-mode
        batch = _build_batch(anat_dir, anat_gm_dir, model.input_size)
        if batch is None or batch.shape[0] == 0:
            return None

        with torch.no_grad():
            logits = model(batch)                 # (N,)
            probs = torch.sigmoid(logits)         # per-slice ADHD probability

        # Patient-level aggregation is the SERVING layer's job (the CNN is a pure
        # per-slice classifier): mean slice probability -> one 0-100 risk.
        risk = float(probs.mean().item()) * 100.0
        # Confidence = how decisively slices sit away from the 0.5 boundary
        # (0 = model on the fence, 100 = fully confident), averaged over slices.
        confidence = float((probs - 0.5).abs().mean().item()) * 2.0 * 100.0

        return {
            "mri_risk_score": round(min(max(risk, 0.0), 100.0), 2),
            "confidence": round(min(max(confidence, 0.0), 100.0), 2),
            "slices_scored": int(batch.shape[0]),
        }
    except Exception as exc:
        # Expected when the on-disk weights don't match the current architecture
        # (e.g. after a model change) or the file is unreadable. Degrade cleanly to
        # "pending" with a concise, actionable line - not an alarming stack trace.
        logger.warning(
            "MRI weights at %s could not be loaded (%s); returning 'pending'. "
            "If the model architecture changed, retrain with train_mri.py to "
            "regenerate a compatible mri_cnn.pt.", CNN_PATH, exc)
        return None


# =============================================================================
# Core: connect, fetch current scans, score, persist, return
# =============================================================================
def analyze_mri(patient_id: str) -> dict:
    """Score a patient's CURRENT MRI scans with the image classifier and persist.

    Called by the API when an MRI ingestion session completes or a Combined Risk
    Analysis is triggered. Opens its own DB connection, reads the patient's
    is_current anat + anat_gm folders, runs the CNN, writes mri_risk_score back
    onto those MRI rows, and returns the diagnosis + confidence. If no trained
    model exists yet it returns a "pending" status (no fabricated score).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Confirm the patient exists (clear 404 vs "no scans").
        cursor.execute("SELECT 1 FROM dbo.Patient WHERE patient_ID = ?;", patient_id)
        if cursor.fetchone() is None:
            raise HTTPException(
                status_code=404, detail=f"Patient '{patient_id}' not found.")

        # Pull the CURRENT scan folder per type (longitudinal model: is_current=1).
        cursor.execute(
            """
            SELECT MRI_ID, scan_type, file_path
            FROM dbo.MRI
            WHERE patient_ID = ? AND is_current = 1;
            """,
            patient_id,
        )
        rows = cursor.fetchall()
        scans = {scan_type: {"mri_id": mri_id, "dir": file_path}
                 for (mri_id, scan_type, file_path) in rows}

        if "anat" not in scans or "anat_gm" not in scans:
            raise HTTPException(
                status_code=404,
                detail=(f"Patient '{patient_id}' does not have both current "
                        "anat and anat_gm scans. Ingest MRI first."),
            )

        scored = _score_with_cnn(scans["anat"]["dir"], scans["anat_gm"]["dir"])

        # No trained model / unreadable slices -> honest pending result.
        if scored is None:
            return {
                "status": "pending",
                "patient_ID": patient_id,
                "mri_risk_score": None,
                "predicted_diagnosis": None,
                "confidence": None,
                "model_version": MODEL_VERSION,
                "message": ("Image classifier not available yet (no trained "
                            "weights). MRI slices are ingested and ready; run "
                            "the training branch to enable scoring."),
            }

        risk = scored["mri_risk_score"]
        diagnosis = "ADHD" if risk >= MRI_DIAGNOSIS_THRESHOLD else "Control"

        # Persist the image risk onto BOTH current MRI rows (mirrors the DSM-5
        # engine writing nlp_risk_score back onto the assessment).
        for scan in scans.values():
            cursor.execute(
                "UPDATE dbo.MRI SET mri_risk_score = ? WHERE MRI_ID = ?;",
                risk, scan["mri_id"],
            )
        conn.commit()

        return {
            "status": "success",
            "patient_ID": patient_id,
            "mri_risk_score": risk,
            "predicted_diagnosis": diagnosis,
            "confidence": scored["confidence"],
            "slices_scored": scored["slices_scored"],
            "model_version": MODEL_VERSION,
        }

    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        logger.exception("MRI analysis failed.")
        raise HTTPException(status_code=500, detail=f"MRI analysis failed: {exc}")
    finally:
        conn.close()


@router.post("/mri/{patient_id}")
def analyze_mri_endpoint(patient_id: str) -> dict:
    """Run the image classification model for one patient and store the MRI risk."""
    return analyze_mri(patient_id.strip())
