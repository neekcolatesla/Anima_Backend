# Anima — Known limitations & methodology: DSM-5 text & demographic model

This document records a label-leakage problem found in the DSM-5 model's
evaluation, how it was diagnosed and corrected, and a subsequent multimodal
fusion finding. It is written to be cited directly in the dissertation.

---

## 1. Initial result: perfect accuracy (a red flag)

A 5-fold stratified cross-validation of the text-and-demographic model on the
seeded cohort (129 patients; 69 ADHD, 60 control) initially returned **perfect
discrimination** (accuracy = F1 = 1.000, ± 0.000 across folds).

Because the metrics were computed on **held-out** folds, this was not classic
overfitting (which shows high train / low test accuracy). Perfect held-out
performance with zero variance is the signature of **label leakage** — the
features already contained the answer.

## 2. Root cause (ablation)

An ablation study — retraining the same model over the same folds on different
feature subsets — localised the leak:

| Feature set (leaky notes) | Accuracy | F1 |
|---------------------------|----------|----|
| Full (scores + demographics + notes) | 1.000 | 1.000 |
| Scores + demographics only | 0.922 | 0.925 |
| Clinical notes only (Bio_ClinicalBERT) | **1.000** | 1.000 |

The clinical notes **alone** reproduced the perfect score. The synthetic notes
had been generated with the diagnosis stated in plain text (e.g. *"Presentation
is consistent with ADHD – Combined Type"*), and control notes used a separate
template, so the text was trivially separable. A secondary, milder circularity
also exists in the structured features: ADHD-200 derives the DX label from the
same Conners' instruments (Inattentive / Hyper-Impulsive T-scores) supplied as
inputs, which is why scores-only already reaches 0.922.

## 3. Correction: regenerating the notes

The note generator (`scripts/generate_dsm5_dataset.py`) was rewritten so the
notes no longer encode the label:

- the diagnosis/subtype is **never stated**;
- behaviours are mentioned **probabilistically** — a mixing weight blends the
  patient's true symptom severity with a flat noise baseline (default 25% signal
  / 75% noise, `--signal 0.25`);
- ADHD and control notes draw on the **same vocabulary** and include neutral,
  non-diagnostic filler, so the two overlap heavily.

After regeneration, no note contains any diagnosis word, and a TF-IDF probe of
the notes fell from perfect separation to ~0.61.

## 4. Post-correction result

Re-running the cross-validation on the corrected notes:

| Feature set (corrected notes) | Accuracy | F1 |
|-------------------------------|----------|----|
| Scores + demographics only | 0.922 ± 0.035 | 0.925 ± 0.036 |
| Clinical notes only (Bio_ClinicalBERT) | 0.611 ± 0.087 | 0.652 ± 0.093 |
| Full (scores + demographics + notes) | 0.713 ± 0.056 | 0.726 ± 0.066 |

The leakage is resolved: notes-only (0.611) now sits just above the majority-class
baseline (0.535), i.e. the text carries only weak signal, as designed.

## 5. Multimodal fusion finding

A second issue surfaced: the **full model (0.713) scores below scores-only
(0.922)** — adding the notes *degraded* the model. This is a curse-of-
dimensionality effect. The feature vector is 5 informative structured dimensions
plus 768 weak note dimensions; once all 773 are standardised to equal variance, a
linear model trained on only ~130 samples lets the 768 low-signal dimensions
collectively outweigh the 5 strong ones, pulling the combined prediction down
toward the notes-only score.

## 6. Mitigation (fusion experiment)

Two principled mitigations were added and compared over identical folds via
`python train_dsm5.py --experiment`:

- **Stronger L2 regularisation** (`--weight-decay` sweep) to shrink the noisy
  note weights.
- **Note-embedding PCA** (`--note-pca`) to project the 768-d embedding to a small
  number of components before fusion, fitted per training fold and baked into the
  model (so serving still uses the raw 773-d vector).

Results (5-fold CV, identical folds, ranked by F1):

| Configuration | Accuracy | F1 |
|---------------|----------|----|
| scores + demographics | **0.922 ± 0.035** | **0.925 ± 0.036** |
| full — PCA-8 (wd = 0.01) | 0.899 ± 0.041 | 0.904 ± 0.043 |
| full — PCA-16 (wd = 0.01) | 0.875 ± 0.065 | 0.877 ± 0.072 |
| full — PCA-32 (wd = 0.01) | 0.806 ± 0.033 | 0.816 ± 0.025 |
| full — naive (wd = 0.1) | 0.752 ± 0.018 | 0.764 ± 0.009 |
| full — naive (wd = 0.5) | 0.736 ± 0.045 | 0.751 ± 0.051 |
| full — naive (wd = 1.0) | 0.721 ± 0.058 | 0.734 ± 0.071 |
| full — naive (wd = 0.01) | 0.713 ± 0.056 | 0.726 ± 0.066 |
| notes only | 0.611 ± 0.087 | 0.652 ± 0.093 |

Two findings follow. First, **the mitigation works**: projecting the 768-d note
embedding to a small number of PCA components restores the full model from 0.713
(naive fusion) to **0.904** (PCA-8), essentially matching the scores-only baseline
within cross-validation variance. The monotonic trend PCA-8 > PCA-16 > PCA-32 is
consistent with the notes carrying little genuine signal — the more note
dimensions retained, the more noise is reinjected — and weight decay alone gave
only partial recovery (up to 0.764 F1). Second, and more important for
interpretation, **no fusion configuration exceeds the scores-only baseline**. On
this dataset the text modality therefore adds no independent predictive value;
the most fusion can achieve is to be absorbed without degrading the structured
features. The shipped model is the best full configuration (PCA-8), which keeps
the hybrid architecture while remaining robust; a structured-only model would be
marginally higher but discards the text component entirely.

## 7. Overall limitation

The ingestion, feature-extraction, cross-validation, fusion, and serving pipeline
are demonstrated to function end-to-end. However, because the clinical notes are
**synthetic and derived from the label**, any signal they carry is a property of
the generator's design, not evidence of real-world NLP performance. The most
defensible reported figure is the structured-feature baseline (~0.92), with the
circularity caveat of Section 2. Establishing the true predictive value of the
natural-language component would require **independently-authored clinical
narratives** and, ideally, **external validation on a held-out cohort**. These
are identified as future work.
