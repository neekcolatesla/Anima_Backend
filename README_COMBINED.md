# Anima — Combined Risk Analysis

The orchestration layer that turns Anima's two machine-learning channels into one
ADHD decision for the clinician. It fuses the **text & demographic model**
(`DSM5_Analysis`) and the **image classification model** (`MRI_Analysis`) into a
single `final_combined_score` + subtype, and records every run in an append-only
audit table.

`app/Combined_Analysis.py` → `POST /api/analysis/combined/{patient_id}`.

## What one call does

1. **Runs both engines through their own code** — `DSM5_Analysis.analyze_patient`
   and `MRI_Analysis.analyze_mri`. This module never re-implements their scoring;
   it only orchestrates and fuses (the same separation-of-concerns the model /
   serving / training split follows elsewhere).
2. **Fuses the two risks** (each 0–100) into `final_combined_score` with a weighted
   average. The text/demographic channel is the stronger ADHD signal, so it is
   weighted higher by default (`COMBINED_NLP_WEIGHT` 0.7, `COMBINED_MRI_WEIGHT` 0.3).
3. **Carries the subtype** — a single aggregate score can't separate the
   presentations, so `predicted_subtype` (Inattentive / Hyperactive / Combined /
   None) comes from the DSM-5 subscale logic.
4. **Diagnoses** — `ADHD` if the combined score ≥ `COMBINED_DIAGNOSIS_THRESHOLD`
   (default 50), else `Control`.
5. **Writes one `Analysis_Result` row** — the authoritative, never-updated audit
   trail for the Admin accountability persona. Each run appends a new row (patient,
   assessment_ID, MRI_ID, both component scores, the combined score, subtype,
   `model_version`, `created_by`); previous runs are preserved.

## Graceful degradation

The DSM-5 assessment is **required** — a patient with none returns 404. The image
model is **optional**: if it has no trained weights yet (`"pending"`) or the
patient has no current MRI scans (`no_scans`), the combined score falls back to the
text/demographic risk alone, records that in `weighting` and `mri_status`, and the
audit row stores `mri_risk_score = NULL`. The analysis never fails just because
imaging isn't ready — which is why the platform is useful before the MRI model is
trained, and why (given structural MRI is a weak ADHD channel) the strong DSM-5
signal carries the headline accuracy.

## Response shape

```json
{
  "status": "success",
  "patient_ID": "0010001",
  "nlp_risk_score": 80.0,
  "mri_risk_score": 40.0,
  "mri_status": "success",
  "final_combined_score": 68.0,
  "predicted_diagnosis": "ADHD",
  "predicted_subtype": "Inattentive",
  "weighting": "nlp=0.7, mri=0.3",
  "model_version": "combined_v1"
}
```

When imaging isn't ready, `mri_risk_score` is `null`, `mri_status` is `pending`
or `no_scans`, and `weighting` reads `nlp=1.0 (mri unavailable)`.

## Configuration (env vars, all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `COMBINED_NLP_WEIGHT` | `0.7` | weight on the text/demographic risk |
| `COMBINED_MRI_WEIGHT` | `0.3` | weight on the image risk |
| `COMBINED_DIAGNOSIS_THRESHOLD` | `50` | combined risk (0–100) at/above which → `ADHD` |
| `COMBINED_MODEL_VERSION` | `combined_v1` | version tag recorded on each audit row |

## Try it

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/analysis/combined/0010001"
```

## Where the accuracy comes from

The DSM-5/demographic model is the strong channel (scores-only cross-validated
~0.92; see `README_DSM5.md` / `LIMITATIONS.md`), while structural-MRI-only ADHD
classification is a hard problem that sits near baseline at this sample size (see
`README_MRI.md`). The default NLP-heavy weighting reflects that reality: the
combined score is where a clinically useful accuracy is realistic, with imaging
as a supporting contributor rather than the driver. Tune the weights via the env
vars above if you want to explore the trade-off.
