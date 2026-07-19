# Anima — DSM-5 model training (text & demographic model)

Trains the ADHD-risk head for the text & demographic model using 5-fold
stratified cross-validation over the seeded database, and saves the best fold's
weights where the API loads them. Runs on Windows PowerShell from your `.venv`;
no Linux required.

This is step two of the hybrid NLP model: `dsm5_features.py` turns each patient
into a feature vector (frozen Bio_ClinicalBERT embedding + demographics), and the
script here trains a small classifier head on top of those features. XAI /
explainability is intentionally not included yet.

---

## 1. Prerequisites

- The Anima database is up and **seeded** — see `README_DB_SETUP.md`. The trainer
  reads the seeded `Patient` + `DSM5_Assessment` rows.
- Your `.venv` (kept **outside** OneDrive, e.g. `C:\venvs\anima`) with
  `requirements.txt` installed, plus **scikit-learn**:

  ```powershell
  C:\venvs\anima\Scripts\Activate.ps1
  pip install -r requirements.txt      # scikit-learn is now included
  ```

- `.env` configured with `DB_HOST=localhost` (running from the host `.venv`) and a
  stable `FERNET_KEY` — the same one used to seed, so encrypted fields line up.
- Internet access on first run: Bio_ClinicalBERT (~440 MB) downloads once from
  Hugging Face and is then cached under your user profile for all later runs.

### Pre-downloading the model & offline mode

By default the app contacts Hugging Face on every startup to check the cache
(even though the weights are already local), so startup depends on the network.
To make it fully offline and faster — recommended for demos on flaky wifi, and
required on locked-down machines — pre-download the model once, then switch to
offline mode.

```powershell
# 1. cache the model (run once, while online; a no-op if already cached)
python scripts\predownload_model.py
```

```ini
# 2. add to .env  (already set on the dev machine)
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

With those flags set, `train_dsm5.py`, `dsm5_smoketest.py`, and the API load
Bio_ClinicalBERT straight from the local cache with **no network calls**. `.env`
is read before transformers is imported, so the flags take effect automatically.

**Verifying it works.** The training/test scripts load the model immediately, so
the `https://huggingface.co/...` request lines just won't appear. The API loads
the model **lazily on the first** `POST /api/analysis/dsm5/{id}` call (not at
startup), so a clean startup log alone doesn't prove it — score one patient and
check the response:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/analysis/dsm5/0010001"
```

A response with `"scoring_method": "model"` (and a risk score identical to an
online run) confirms the model loaded from the cache — had the offline load
failed, the endpoint would have fallen back to `"scoring_method": "heuristic"`.
In the server logs you'll see `Loading clinical language model ...` →
`Clinical language model loaded.` with **no** Hugging Face request lines between
them.

The pre-download script forces online mode for itself, so it works even with the
flags set. Run it **before** relying on offline mode: if offline startup ever
errors that it cannot find or reach the model, a needed file was not cached —
re-run `predownload_model.py` online, or temporarily remove the two flags.

For the **Docker image this is already wired**: the `Dockerfile` runs
`predownload_model.py` during the build (baking the model into the image's HF
cache at `HF_HOME=/opt/hf-cache`) and sets `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1`, so the containerised API loads the model from the baked
cache with no network at runtime. The one-time download happens during
`docker compose build`, which needs internet.

---

## 2. Run it

```powershell
docker compose up -d db      # if the DB container isn't already running
cd app
python train_dsm5.py
```

The first run is slow while the language model downloads and embeds the notes;
subsequent runs are fast. No arguments are needed — the defaults reproduce the
evaluated run.

---

## 3. What it does (pipeline)

| Step | Action |
|------|--------|
| 1 | Connects to SQL Server via the existing `.env` / `database.py`. |
| 2 | Pulls `Patient` + `DSM5_Assessment` rows with a **confirmed diagnosis** (`ground_truth_dx IS NOT NULL`). Controls (DX = 0) are included — a binary ADHD-vs-control classifier needs both sides. |
| 3 | Runs every row through `dsm5_features.build_feature_matrix` → one `(N, 773)` tensor. This is the **same** feature function the API uses at inference, so there is no train/serve skew. |
| 4 | Builds `dsm5_model.DSM5Head` — a single linear layer (773 → 1 logit → sigmoid ADHD-risk). The exact class the API loads. Kept deliberately tiny to learn signal rather than memorise ~130 rows. |
| 5 | Splits with **5-fold stratified cross-validation** (see below). |
| 6 | Standard PyTorch train/eval loop, run once per fold (5 times). |
| 7 | Prints **accuracy + F1** after each fold. |
| 8 | Tracks the fold with the highest F1. |
| 9 | Saves that winning fold's weights to `app/models/dsm5_head.pt`. |
| + | Runs an **ablation** (on by default): the same folds, retrained on feature subsets, to show where the signal comes from. |

### The label

The learned model is **binary**: ADHD (DX ∈ {1, 2, 3}) vs control (DX = 0),
matching the single-logit endpoint that produces `nlp_risk_score`. The ADHD
**subtype** is decided separately by the rule-based logic already in
`DSM5_Analysis.py` (which subscale T-scores are clinically elevated), not by this
model.

### Why the folds are built the way they are

The seeded set (patients 1–129) is:

| DX | Subtype | Count |
|----|---------|-------|
| 0 | Control | 60 |
| 1 | ADHD-Combined | 39 |
| 3 | ADHD-Inattentive | 28 |
| 2 | ADHD-Hyperactive/Impulsive | **2** |

The hyperactive-only subtype has just **2** patients. A plain
`StratifiedKFold(5)` on the 4-class subtype is invalid (it requires ≥ 5 members
per class) and, even when forced, can drop both rare cases into a single fold. So
the trainer:

- splits the common subtypes (Control, Combined, Inattentive) with sklearn
  `StratifiedKFold` — proportional, reproducible, recognised;
- allocates the rare hyperactive cases **round-robin across folds**, which
  guarantees the two land in **different** test folds.

Each fold prints its test-set subtype composition so the spread is visible in the
output and can be cited in the write-up. The overall ADHD/control balance
(60 / 69) is close to even, so the binary task is well-posed; the loss still
applies a positive-class weight per fold as a safeguard.

---

## 4. Expected output (shape)

```
Loaded 129 patients (confirmed diagnosis).
  Binary:   ADHD=69  Control=60
  Subtypes: Control=60  ADHD-Combined=39  ADHD-Inattentive=28  ADHD-Hyper/Imp=2
Building features via Bio_ClinicalBERT (first run downloads the model) ...
Feature matrix: (129, 773)
  fold alloc: DX=2 (ADHD-Hyper/Imp, n=2) round-robin across folds ...

=== 5-fold stratified cross-validation ===
Fold 1/5 | test n=26 | accuracy=... | F1=... | test subtypes: Control=12  ADHD-Combined=8  ADHD-Inattentive=6  ADHD-Hyper/Imp=1
... (folds 2–5) ...

=== Cross-validation summary ===
Accuracy: 0.xxx +/- 0.0xx
F1:       0.xxx +/- 0.0xx
Best fold: #k (F1=..., accuracy=...)

Saved winning model weights -> ...\app\models\dsm5_head.pt
```

(The exact numbers depend on the trained embeddings; the structure is fixed.)

---

## 4a. Ablation: reading the results honestly

On this dataset a near-perfect ADHD-vs-control score is **expected, and not a
sign of a clever model** — it reflects two forms of label leakage:

- The Conners T-scores (`inattentive_score`, `hyperactive_score`) are part of the
  feature vector, but in ADHD-200 the DX label is *derived from* those very
  instruments. Those two numbers alone give ~92-93% CV accuracy.
- The clinical notes were **synthetically generated from the diagnostic scores**
  (`scripts/generate_dsm5_dataset.py`), so their Bio_ClinicalBERT embedding also
  encodes the label.

Perfect accuracy here is measured on *held-out* folds, so it is leakage, not
classic overfitting (overfitting would show high train / low test). The trainer
therefore runs an ablation by default — the same folds retrained on:

    full (scores + demographics + notes)
    scores + demographics only
    clinical notes only (BERT)

and prints a comparison table. Interpret it like this:

- **notes-only ≈ full** → the synthetic notes are leaking the label (the text
  isn't adding independent clinical insight, it's echoing the diagnosis).
- **full ≤ scores-only** → the text adds no signal beyond the instrument scores.

This is the honest framing for the write-up: the pipeline is demonstrated
end-to-end, the clinical scores are legitimately (if circularly) predictive, and
the text model's apparent performance is bounded by the fact that the notes are
label-derived. Independently-written clinical notes would be needed to evaluate
the NLP component's true predictive value. Disable the ablation with
`--no-ablation` if you only want the shipped model.

---

## 5. How it plugs into the API

The winning weights are saved to `app/models/dsm5_head.pt` — the exact path
`DSM5_Analysis.py` reads from. On the next call to
`POST /api/analysis/dsm5/{patient_id}`, the endpoint rebuilds `DSM5Head`, loads
these weights via `dsm5_model.load_head()`, and scores with the trained model
instead of the transparent heuristic fallback. Nothing else needs wiring.

The artifact is a `{config, state_dict}` dict of tensors + primitives (the config
records the architecture, e.g. the note-PCA size), so `load_head()` rebuilds the
exact model and it loads safely under the `weights_only=True` default of torch ≥ 2.6.

---

## 6. Reproducibility & configuration

The run is deterministic given the seed (NumPy + torch seeded; the frozen
encoder runs in eval mode with no gradients). Useful flags:

```powershell
python train_dsm5.py --folds 5 --epochs 300 --lr 0.01 --weight-decay 0.01 --seed 42
python train_dsm5.py --output ..\dsm5_head.pt      # save somewhere else
```

`--weight-decay` is the main overfitting control — higher values pull the linear
head toward a simpler solution. `--epochs` and `--lr` tune the full-batch Adam
loop.

### Fusion experiment (why the notes can hurt)

The full feature vector is 5 strong structured features + a 768-d note embedding.
On ~130 samples those 768 dimensions can *swamp* the 5, so naive concatenation
may score **below** the scores-only baseline. Two mitigations are built in:

- `--weight-decay 0.1` (or higher) — shrinks the many noisy note weights.
- `--note-pca 16` — projects the 768-d embedding to 16 PCA dims (fitted per fold,
  baked into the model, so the API still feeds raw 773-d vectors) before fusion.

Run the whole comparison at once and let it pick + save the best full config:

```powershell
python train_dsm5.py --experiment
```

This sweeps weight decay (0.01 → 1.0) and note-PCA (8/16/32) over the same folds,
prints a ranked table, and saves the best-F1 **full** configuration to
`app/models/dsm5_head.pt`. Read it as: if a fusion config matches or beats
scores-only, the notes can be combined without degradation; if only scores-only
wins, the (synthetic) text carries no usable independent signal here.

---

## 7. Testing the trained model

Two ways to exercise the trained model against the **seeded 1–129 patients**. The
held-out 130–222 block is never touched by either (it was never seeded), so it
stays sealed for the moderated sessions.

> These are **functional smoke tests** — "does the trained head load and return
> sensible outputs through the real endpoint code?" — not a performance
> measurement. The model was trained/cross-validated on these same patients, so
> scoring them again is optimistic by construction; the honest metrics are the
> 5-fold CV numbers above and in `LIMITATIONS.md`.

### 7a. Batch smoke test — `dsm5_smoketest.py`

Runs every seeded patient through the real `DSM5_Analysis.analyze_patient()`
path (the exact code the endpoint calls): loads the trained head, builds the
features, scores risk, predicts subtype, and writes `nlp_risk_score` back to the
DB.

```powershell
cd app
python dsm5_smoketest.py              # all 129 seeded patients
python dsm5_smoketest.py --limit 15   # quicker sample
python dsm5_smoketest.py --patient 0010001   # a single patient
```

What to check in the summary:

- `Used trained model: 129/129` — confirms the trained PCA head is in use (not
  the heuristic fallback). Any `heuristic` rows mean the head didn't load.
- `Mean risk – ADHD` clearly above `Mean risk – Control` — the model learned the
  right direction.
- The architecture line reports the shipped config, e.g. `PCA-8 fusion`.

### 7b. Live API endpoint

Start the server from `app/` (with the DB up and `.env` set to
`DB_HOST=localhost`):

```powershell
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Score patients from a second terminal — use real seeded IDs (`0010001`–`0010129`):

```powershell
foreach ($id in '0010001','0010005','0010016','0010049') {
  Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/analysis/dsm5/$id"
}
```

Each call returns `{ status, patient_ID, nlp_risk_score, predicted_subtype,
scoring_method }`. The first request is slow (Bio_ClinicalBERT loads into memory
once), then subsequent calls are fast. Or open **http://localhost:8000/docs**,
expand `POST /api/analysis/dsm5/{patient_id}`, and use **Try it out** — this
Swagger page is the same endpoint the PowerApps frontend calls and is the
cleanest thing to show in a demo.

> **Offline / demo tip:** make startup network-independent and faster by
> pre-downloading the model and enabling offline mode — see
> "Pre-downloading the model & offline mode" under Prerequisites.

---

## 8. Troubleshooting

- **`ModuleNotFoundError: sklearn`** — `pip install scikit-learn` in the active
  `.venv` (it's in `requirements.txt`; reinstall if your venv predates it).
- **Hugging Face download fails / offline** — the first run needs internet to
  fetch Bio_ClinicalBERT; after that it's cached. Run `scripts/predownload_model.py`
  once while online, then set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in
  `.env` (see "Pre-downloading the model & offline mode" above).
- **`No labelled patients` / count is 0** — the database isn't seeded; run
  `python seed_dsm5.py` first (see `README_DB_SETUP.md`).
- **`OSError [Errno 22]` importing numpy/torch** — the `.venv` is inside OneDrive;
  recreate it on a local path such as `C:\venvs\anima` (see `README_DB_SETUP.md`).
- **Slow first run** — expected; it's the one-time model download + embedding
  129 clinical notes. Later runs reuse the cache.
- **`UNEXPECTED` keys in a BertModel load report** (`cls.predictions.*`,
  `cls.seq_relationship.*`) — **expected and harmless**. Bio_ClinicalBERT ships
  with a masked-language-model head that is discarded when the model is loaded as
  a plain encoder (`AutoModel`); those are the discarded head weights, not a
  problem with the download or the model.
