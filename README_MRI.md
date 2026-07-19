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
| Training script | `train_dsm5.py` (built) | `train_mri.py` (built) |

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
2. **Select + pair** the slices via the shared `mri_slices` module — the *same*
   selector the trainer uses, so serving scores the same slices training saw (no
   skew). It keeps the informative central region of the stack (`MRI_CENTRAL_FRAC`,
   default 0.6), optionally drops near-empty slices (`MRI_MIN_FOREGROUND`), pairs
   the two stacks positionally, resizes each to the model's `input_size` (128),
   scales to `[0, 1]`, and stacks as a **2-channel** batch `(N, 2, 128, 128)` —
   channel 0 anat, channel 1 anat_gm.
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
| `MRI_CENTRAL_FRAC` | `0.6` | keep the middle fraction of the axial stack (skip empty edges) — **must match training** |
| `MRI_MIN_FOREGROUND` | `0.0` (off) | drop slices with too little brain tissue — **must match training** |
| `MRI_MAX_SLICES` | `0` (all) | cap selected slices scored per patient (evenly sampled) — bounds serve-time cost |

> The two selection vars must match whatever the shipped `mri_cnn.pt` was trained
> with, or serving scores a different slice set than the model learned on. If you
> retrain with non-default `--central-frac` / `--min-foreground`, set the matching
> env vars for the API too.

### Try it

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/analysis/mri/0010001"
```

Before the model is trained this returns `"status": "pending"` (expected). After
training saves `mri_cnn.pt`, the same call returns a `"success"` result with a
risk score and diagnosis.

## 3. Training — `app/train_mri.py`

Trains `MRICNN` and saves the best fold's weights to `app/models/mri_cnn.pt` —
exactly where `MRI_Analysis.py` loads them, so a successful run flips the endpoint
from `"pending"` to `"success"` automatically. It imports the CNN from
`mri_model.py` and trains **only** that shallow architecture (no deeper model), so
it learns real anatomical differences rather than memorising ~130 patients. This
parallels `train_dsm5.py` (see `README_DSM5.md`).

What one run does:

1. **Labels** — pulls each patient's true `ground_truth_dx` from
   `DSM5_Assessment` (binary target: ADHD = DX ∈ {1,2,3}, control = 0).
2. **Images** — finds each patient's **current** (`is_current = 1`) `anat` and
   `anat_gm` folders in the `MRI` table; a patient needs a label **and** both
   scans to be included (others are skipped with a count).
3. **Samples** — selects + pairs slices via the shared `mri_slices` module (the
   *same* code the API serves with): keep the informative central region
   (`--central-frac`, default 0.6), optionally drop near-empty slices
   (`--min-foreground`), pair positionally, and stack each pair into a 2-channel
   `(2, H, W)` sample (channel 0 anat, channel 1 anat_gm). Slices are loaded
   **lazily from disk** by a `Dataset`, so the full slice set never sits in RAM.
4. **Split by patient, not by slice** — `StratifiedKFold(5)` runs over **patients**.
   If a patient is in the test fold, **all** of their slices are in test — so no
   slice from one brain ever appears in both train and test. The loop asserts zero
   patient overlap per fold as a safety net.
5. **Per-slice training** — the CNN outputs one logit per slice; it trains on
   slices with `BCEWithLogitsLoss` (positive class weighted for ADHD/control
   imbalance), each slice carrying its patient's label.
6. **Per-fold accuracy** — after each fold it prints **patient-level** accuracy +
   F1 (the deployment metric — slices are aggregated by mean probability exactly
   as `MRI_Analysis.py` serves) and **slice-level** accuracy as a lower-level check.
7. **Best fold** — tracks the fold with the best patient-level accuracy (tie-broken
   by F1) and, on finish, saves its weights via `mri_model.save_cnn`.

```powershell
python train_mri.py                                  # 5-fold CV, central slices
python train_mri.py --epochs 15 --max-slices 60      # faster / lighter run
python train_mri.py --central-frac 0.5 --min-foreground 0.05   # tighter slice selection
python train_mri.py --sweep                          # exploratory tuning grid
```

Key options: `--folds` (default 5), `--epochs` (12), `--lr` (1e-3),
`--weight-decay` (1e-4), `--batch-size` (64), `--input-size` (128), `--dropout`
(0.4), `--central-frac` (0.6, middle fraction of the stack kept), `--min-foreground`
(0.0, drop slices with too little tissue), `--max-slices` (0 = all selected;
evenly sampled — the training analogue of serving `MRI_MAX_SLICES`), `--threshold`
(0.5, the mean-probability cut-off), `--num-workers` (0 — safest on Windows),
`--seed`, `--output`. Runs on GPU automatically if `torch.cuda.is_available()`.

> **If you change `--central-frac` / `--min-foreground`, set the matching
> `MRI_CENTRAL_FRAC` / `MRI_MIN_FOREGROUND` env vars for the API**, so it scores
> the same slice selection the shipped weights were trained on.

**Why two accuracies?** The CNN is a per-slice classifier, but the patient is what
gets diagnosed. Patient-level accuracy (mean-aggregated, matching serving) is the
honest test score and the fold-selection metric; slice-level accuracy is a sanity
signal underneath it.

### Tuning — `--sweep`

`--sweep` runs a small curated grid (epochs / lr / regularisation / input size)
over the **same** folds, prints a ranking by mean patient accuracy, and saves the
best. It is **exploratory**: because the grid is scored on the same held-out folds
it is chosen against, treat any "win" cautiously — a real gain should survive a
fresh `--seed`, not just top the table.

### On the expected result (be honest)

Classifying ADHD from *structural* MRI is genuinely hard: on ADHD-200-scale data,
image-only structural models typically land from near-chance to the low-60s percent,
well below the demographic/DSM-5 signal. A near-chance patient accuracy here is
therefore **consistent with the literature, not a bug** — and, crucially, a
chance-level score under a strict patient-level split is what a *non-leaking*
pipeline looks like (the opposite of the DSM-5 notes-leakage failure). The central
slice selection and `--sweep` are the principled levers to probe for signal; they
are not guaranteed to lift a fundamentally weak modality, and should not be pushed
into overfitting the test folds.

## 4. What's next (deferred)

- **Combined Risk Analysis** — the orchestration that fuses `nlp_risk_score` and
  `mri_risk_score` into `Analysis_Result.final_combined_score` + subtype (the
  append-only audit table already exists for it).
