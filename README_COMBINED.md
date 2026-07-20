# Anima — Combined Risk Analysis

The orchestration layer that turns Anima's two machine-learning channels into one
ADHD decision for the clinician. It fuses the **text & demographic model**
(`DSM5_Analysis`) and the **image classification model** (`MRI_Analysis`) into a
single `final_combined_score` + subtype, and records every run in an append-only
audit table.

`app/Combined_Analysis.py` → `POST /api/analysis/combined/{patient_id}?requester_user_id=...`.

The trigger is **single-patient** (one analysis at a time — no batch), RBAC-gated
by the same rules as the read side below, and records the requester as
`created_by` on the audit row.

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
   `model_version`, and `created_by` = the `requester_user_id` that triggered it);
   previous runs are preserved.

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

## Retrieval (read side) — the clinician view & Admin audit trail

Two RBAC-gated GET endpoints read back the append-only `Analysis_Result` table
(the write side is the POST above). Like the rest of the API, the caller passes
`requester_user_id` as a query parameter.

| Endpoint | Returns |
|----------|---------|
| `GET /api/analysis/results/{patient_id}` | the patient's **latest** combined result |
| `GET /api/analysis/results/{patient_id}/history?limit=50` | the patient's **full history**, newest first (the audit trail) |

**Access control** (enforced by `_authorize_patient_access`, and this also protects
child records — a child has no login, so the only ways in are Admin, its assigned
clinician, or its linked guardian):

| Role | May read |
|------|----------|
| **Admin** | any patient (the accountability / audit persona) |
| **Clinician** | only patients assigned via `Clinician_Patient_Assignment` |
| **Guardian** | only their linked child (`Patient.guardian_ID`) |
| **Patient** | only themselves (`Patient.user_ID`) |

Denials return `403`; an unknown requester `401`; a patient with no results `404`.

> The seeded database ships an Admin (so the audit view works out of the box) but
> **no clinician assignments** — grant one to test the clinician path, e.g.
> `INSERT INTO dbo.Clinician_Patient_Assignment (clinician_ID, patient_ID) VALUES ('0010005', '0010001');`

## Try it

```powershell
# Run the analysis (RBAC-gated; records created_by = requester)
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/analysis/combined/0010001?requester_user_id=A000001"

# Read it back (Admin sees any patient)
Invoke-RestMethod -Uri "http://localhost:8000/api/analysis/results/0010001?requester_user_id=A000001"
Invoke-RestMethod -Uri "http://localhost:8000/api/analysis/results/0010001/history?requester_user_id=A000001"
```

Or validate both sides against the DB without the server running:

```powershell
python combined_smoketest.py --limit 5 --as A000001        # write side (+ created_by)
python combined_read_smoketest.py --patient 0010001         # read side (+ RBAC denial check)
```

## Explainability (XAI) — the `explanation` block

Every successful analysis now carries an `"explanation"` object so the frontend can
show **why** the model decided as it did. The DSM-5 endpoint returns `text_model`;
the Combined endpoint returns both `text_model` and `mri_model`; the MRI endpoint
returns `mri_model`.

```jsonc
"explanation": {
  "text_model": {
    "method": "model",
    "summary": "The biggest factor was 'Hyperactive score' (38.1% toward ADHD). Notable words in the note: restless, distracted.",
    "feature_importance": [        // bind to a bar chart
      { "feature": "Hyperactive score", "impact_percent": 38.1, "direction": "toward ADHD" },
      { "feature": "Clinical notes",    "impact_percent": 22.4, "direction": "toward ADHD" },
      { "feature": "Inattentive score", "impact_percent": 19.0, "direction": "toward ADHD" }
      // ... Age, Biological sex, Child
    ],
    "influential_words": [         // bind to a gallery / list
      { "word": "restless",   "push": 100.0, "direction": "toward ADHD" },
      { "word": "focused",    "push": -61.0, "direction": "toward control" }
    ]
  },
  "mri_model": {
    "available": true,
    "top_slice_index": 94,         // the single highest-risk brain slice
    "heatmap_image": "data:image/png;base64,iVBORw0KGgo...",   // set as an Image control's Image
    "summary": "The image model's decision was driven most by brain slice 94. Red = looked hardest, blue = ignored."
  }
}
```

What each piece is, in plain terms:

- **`feature_importance`** — how much each DSM-5 score / demographic (plus the note
  as one bar) pushed the prediction, as percentages that sum to 100. Because the
  text head is a linear model, these are the model's *exact* contributions, not an
  approximation. `direction` says which way it pushed.
- **`influential_words`** — the note's most impactful words. `push` is that word's
  pull on the score relative to the strongest word (−100…100); positive = toward
  ADHD. Built by dotting each word's language-model vector with the model's note
  weight.
- **`mri_model.heatmap_image`** — a Grad-CAM overlay on the highest-risk slice: the
  script finds the slice with the top ADHD score, traces which pixels drove it, and
  paints a red(=looked hard)/blue(=ignored) map over the brain. It's a base64 PNG
  **data URI**, so in Power Apps you can bind an Image control's `Image` property
  straight to it — no file download needed.

**Power Apps binding sketch:**

- bar chart items → `response.explanation.text_model.feature_importance`
  (X = `feature`, Y = `impact_percent`)
- influential-words gallery → `response.explanation.text_model.influential_words`
- brain heatmap → `Image1.Image = response.explanation.mri_model.heatmap_image`

**Graceful degradation.** If the trained text head can't load, `text_model.method`
is `"heuristic"` (feature importance from the subscale scores, no word list). If the
MRI model is untrained or the patient has no scans, `mri_model` is
`{ "available": false, "reason": "pending" | "no_scans" }` and there's no heatmap —
the rest of the explanation is unaffected.

> Note: this release also fixed the DSM-5 head loader (it now uses
> `dsm5_model.load_head`, so scoring + explanations use the **real trained model**;
> previously a load bug silently fell back to the score-only heuristic).

## Where the accuracy comes from

The DSM-5/demographic model is the strong channel (scores-only cross-validated
~0.92; see `README_DSM5.md` / `LIMITATIONS.md`), while structural-MRI-only ADHD
classification is a hard problem that sits near baseline at this sample size (see
`README_MRI.md`). The default NLP-heavy weighting reflects that reality: the
combined score is where a clinically useful accuracy is realistic, with imaging
as a supporting contributor rather than the driver. Tune the weights via the env
vars above if you want to explore the trade-off.
