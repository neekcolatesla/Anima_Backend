# Anima — MRI image classification engine

The image side of Anima's two-model design: a shallow 2D CNN that classifies a
patient's anatomical MRI slices as **ADHD vs control**, and the serving code that
runs it for one patient and returns a diagnosis + confidence.

This mirrors the DSM-5 text model exactly — **architecture** in one module,
**inference/serving** in another — so the two never entangle:

| Concern | Text & demographic model | Image classification model |
|---------|--------------------------|----------------------------|
| Architecture (shared by train + serve) | `app/dsm5_model.py` (`DSM5Head`) | `app/mri_model.py` (`MRICNN`) |
| Inference engine + API endpoint | `app/DSM5_Analysis.py` → `POST /api/analysis/dsm5/{id}` | `app/MRI_Analysis.py` → `POST /api/analysis/mri/{id}` |
| Saved weights | `app/models/dsm5_head.pt` | `app/models/mri_cnn.pt` |
| Training script | `train_dsm5.py` (built) | *(next branch — not built yet)* |

XAI / explainability is intentionally **not** included yet.

---

## 1. The model — `app/mri_model.py`

A deliberately **shallow** 2D CNN. Medical cohorts are tiny (here ~130 labelled
patients); a deep, high-capacity network would just memorise those volumes and
generalise poorly. So the network is kept light on purpose:

- **three** convolutional blocks only — `Conv2d(3×3, pad 1) → BatchNorm2d → ReLU
  → MaxPool2d(2)`, widths **16 → 32 → 64**;
- a **Global Average Pool** (`AdaptiveAvgPool2d(1)`) instead of a large flattened
  dense head — this collapses each feature map to one number and is what keeps the
  parameter count tiny (no flatten→dense parameter explosion);
- **Dropout** (default 0.4) then a single **`Linear(64 → 1)`** producing one ADHD
  logit; `sigmoid(logit)` = per-slice ADHD probability.

Total ≈ **24k parameters** — enough to learn gross grey-matter morphometric
differences, small enough to resist memorising the set. BatchNorm keeps training
stable on the small batches medical data forces.

**Input contract.** A batch of 2-channel slices, shape `(N, in_channels, H, W)`:

- **channel 0 = `anat`** (warped, skull-stripped structural T1 — whole-brain anatomy),
- **channel 1 = `anat_gm`** (smoothed grey-matter segmentation — where ADHD
  differences concentrate).

`forward()` returns one logit **per slice**, shape `(N,)`. **Per-patient
aggregation is deliberately NOT in the model** — the CNN is a pure per-image
classifier; the serving layer aggregates the slice stack into one patient score.

**Persistence** (same format as the DSM-5 head, `torch.load`-safe under torch ≥ 2.6):

```python
mri_model.save_cnn(model, "app/models/mri_cnn.pt")   # {config, state_dict}
model = mri_model.load_cnn("app/models/mri_cnn.pt")  # rebuilds + eval() mode
```

`config()` records `{in_channels, input_size, dropout, channels}` so `load_cnn`
reconstructs the exact geometry the weights were trained with — no train/serve
skew. `load_cnn` also accepts a bare `state_dict` for backward compatibility.

## 2. The prediction engine — `app/MRI_Analysis.py`

Owns `POST /api/analysis/mri/{patient_id}` and the `analyze_mri(patient_id)`
function the API calls when an **MRI ingestion session completes** or a **Combined
Risk Analysis** is triggered. It **never defines a network** — it only
`load_cnn`s the saved weights and calls the model, so architecture and serving
stay completely separate.

What one call does:

1. Look up the patient's **current** scan folders in the `MRI` table
   (`is_current = 1`) — one `anat` and one `anat_gm` directory of JPEG slices.
2. **Pair** the two stacks positionally, resize each slice to the model's
   `input_size` (128), scale to `[0, 1]`, and stack as a **2-channel** batch
   `(N, 2, 128, 128)` — channel 0 anat, channel 1 anat_gm.
3. Run the CNN → per-slice probabilities, then **aggregate to one patient score**:
   - `mri_risk_score` = mean slice probability × 100 (0–100),
   - `confidence` = mean distance of slice probabilities from the 0.5 boundary,
     ×2 ×100 (0 = model on the fence, 100 = fully decisive),
   - `predicted_diagnosis` = `ADHD` if risk ≥ `MRI_DIAGNOSIS_THRESHOLD` (default
     50) else `Control`.
4. Persist `mri_risk_score` back onto **both** current `MRI` rows (mirroring the
   DSM-5 engine writing `nlp_risk_score` onto the assessment), and return the
   result.

**Graceful "pending" until the model is trained.** Until the training branch
produces `app/models/mri_cnn.pt`, there is no honest image score to give — and
unlike the DSM-5 engine, pixels have no transparent hand-rule fallback. So with
no weights the endpoint returns a clear `"status": "pending"` (score `null`, no
DB write) rather than fabricating a number. torch/PIL and `mri_model` are
imported **lazily** inside the scoring path, so a missing dependency degrades the
same way instead of taking the API down.

### Response shapes

Trained model present:

```json
{
  "status": "success",
  "patient_ID": "0010001",
  "mri_risk_score": 63.4,
  "predicted_diagnosis": "ADHD",
  "confidence": 41.2,
  "slices_scored": 189,
  "model_version": "mri_cnn_v1"
}
```

No trained weights yet:

```json
{
  "status": "pending",
  "patient_ID": "0010001",
  "mri_risk_score": null,
  "predicted_diagnosis": null,
  "confidence": null,
  "model_version": "mri_cnn_v1",
  "message": "Image classifier not available yet (no trained weights) ..."
}
```

### Configuration (env vars, all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `MRI_CNN_PATH` | `app/models/mri_cnn.pt` | trained weights location |
| `MRI_MODEL_VERSION` | `mri_cnn_v1` | version tag recorded on scores |
| `MRI_DIAGNOSIS_THRESHOLD` | `50` | risk (0–100) at/above which → `ADHD` |
| `MRI_MAX_SLICES` | `0` (all) | cap paired slices scored per patient (evenly sampled) — bounds serve-time cost |

### Try it

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/analysis/mri/0010001"
```

Before the model is trained this returns `"status": "pending"` (expected). After
training saves `mri_cnn.pt`, the same call returns a `"success"` result with a
risk score and diagnosis.

## 3. What's next (deferred)

- **`train_mri.py`** — trains `MRICNN` on the seeded slice stacks (labels via
  `DSM5_Assessment.ground_truth_dx`), same shallow-model discipline and
  cross-validation spirit as `train_dsm5.py`, saving the best weights to
  `app/models/mri_cnn.pt` with `mri_model.save_cnn`.
- **Combined Risk Analysis** — the orchestration that fuses `nlp_risk_score` and
  `mri_risk_score` into `Analysis_Result.final_combined_score` + subtype (the
  append-only audit table already exists for it).
