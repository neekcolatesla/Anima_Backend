# Anima — Data Preparation

How the raw ADHD-200 data was turned into the two classifier-ready datasets that
drive Anima's models: the **MRI imaging** dataset (for the image classifier) and
the **text & demographic** dataset (for the DSM-5 / NLP model). This document is
the methodology reference for both streams.

---

## 1. Sources and governance

Both streams derive from the **ADHD-200 Consortium, NYU site** data (the phenotypic
table and the NeuroBureau *Athena* preprocessed imaging), released under
**Creative Commons Attribution-NonCommercial (CC BY-NC)** and de-identified
(subjects are anonymised 7-digit ScanDir IDs).

Two principles govern the prepared data:

- **No real PII.** Patient names and clinical notes are **synthetically generated**
  (see §3), never real. The phenotypic values are the public, de-identified
  ADHD-200 measures.
- **Train vs held-out split.** Patients **1–129** form the seed/training set;
  **130–222** are held out (kept off the repo/DB) to demonstrate the pipeline on
  unseen data during moderated evaluation.
- **Raw data is git-ignored.** The MRI scans (`*.nii`, `*.nii.gz`, `*.zip`,
  `data/mri/`) and the CSVs (`*.csv`) are never committed to the public repo; they
  are delivered with the project folder.

---

## 2. MRI imaging data preparation

### 2.1 Raw-dataset cleaning (removing unwanted files)

The Athena preprocessed download ships **far more than the anatomical scans** —
each patient folder contained ~25 files (~180 MB), dominated by resting-state
functional MRI and tissue-segmentation maps that this project does not use (Anima
classifies on **anatomical** scans only). These were stripped with three recursive
Windows `del` passes from the dataset root (`del` flags: `/S` recurse
sub-folders, `/Q` quiet/no prompt, `/F` force):

```bat
del /S /Q /F *_rest_*      REM resting-state fMRI: fALFF, functional connectivity,
                           REM motion params (rp), smoothed/normalised functional
                           REM volumes (sfnwmrda, snwmrda), time-courses (TCs),
                           REM mean functional (wmean), masks  -- the ~36 MB 4-D data
del /S /Q /F *_csf*        REM CSF (cerebrospinal-fluid) tissue-segmentation maps
del /S /Q /F *_wm*         REM white-matter tissue-segmentation maps
```

**Result:** each patient folder is reduced from ~25 files to the anatomical scans
that the ingestion pipeline consumes — a ~90 %+ reduction in footprint. The two
scans kept per patient are:

| File | Meaning | Ingested as |
|------|---------|-------------|
| `wssd<ID>_session_1_anat.nii.gz`     | warped, skull-stripped **structural T1** | `scan_type = anat` |
| `swssd<ID>_session_1_anat_gm.nii.gz` | smoothed warped **grey-matter** segmentation | `scan_type = anat_gm` |

The grey-matter scan matters because ADHD morphometric differences concentrate in
grey matter; the structural T1 gives whole-brain anatomical context.

The cleaned scans are laid out one folder per patient for the seeder:

```
data/mri/NYU_Athena_preproc_1-129/
├── 0010001/  wssd0010001_session_1_anat.nii.gz   swssd0010001_session_1_anat_gm.nii.gz
├── 0010002/  ...
└── 0010129/  ...
```

### 2.2 Ingestion & preprocessing (NIfTI → classifier-ready images)

The shared core (`app/MRI_Ingestion.py` → `_nii_to_slice_stack`, used by both the
`POST /api/ingest/mri` endpoint and the `seed_mri.py` bulk seeder) turns each 3-D
volume into a 2-D image dataset a CNN can consume:

1. Load the NIfTI volume with **nibabel** (dropping any stray 4th dimension).
2. **Normalise intensities volume-wide to 0–255**, so brightness is consistent
   across every slice and patient (raw NIfTI intensities vary widely between
   scans — a CNN needs a common scale).
3. Extract the **entire axial slice stack** — all **189 slices per scan**, not a
   single mid-slice — for whole-brain coverage of the distributed ADHD biomarkers
   (prefrontal cortex, basal ganglia, cerebellum).
4. Rotate each slice to a consistent display orientation and save it as a
   **grayscale JPEG** (`<patient>_<scan>_<index>.jpg`, zero-padded to preserve
   order) into `app/static/mri_images/<patient>/session_<n>/<scan>/`.

Each patient therefore yields **378 slices** (189 × 2 scans); the full 129-patient
set is **48,762 JPEGs**.

### 2.3 Database records and labels

One `MRI` row is written per scan folder (path, `slice_count`, `is_current`,
`scan_session`, `acquired_at`) under the longitudinal model (a patient may hold
multiple scans over time; one is current per type). Supervised **labels are
already in place**: each `MRI` row links by `patient_ID` to
`DSM5_Assessment.ground_truth_dx` — the ADHD-200 diagnosis (0 = control,
1 = combined, 2 = hyperactive, 3 = inattentive) — so the image classifier trains
on the same binary (ADHD vs control) or multiclass target as the text model.

---

## 3. NLP / DSM-5 text & demographic data preparation

The text model consumes the phenotypic scores plus a DSM-5 questionnaire and a
clinical narrative. The ADHD-200 phenotypic table has the scores but **no
questionnaire or notes**, so those were synthesised.

### 3.1 Phenotypic source + synthetic names

Started from the NYU Athena phenotypic CSV (age, biological sex, handedness,
diagnosis `DX`, Conners' ADHD Index / Inattentive / Hyperactive T-scores, IQ, med
status). A **synthetic full name** was generated for each row — biologically
appropriate to the recorded sex and **culturally diverse** — producing
`NYU_Athena_Phenotypic_with_names.csv`. The names are fictional (no real identity);
they exist so the platform can demonstrate name handling, and they are **encrypted
at rest** (Fernet) when loaded into the database.

### 3.2 Synthetic DSM-5 questionnaire + notes

`scripts/generate_dsm5_dataset.py` reverse-engineers a plausible questionnaire and
narrative from each patient's real T-scores:

- **Questionnaire (`raw_answers`)** — the Conners' T-scores (~40–90) are mapped
  proportionally into the 18-item DSM-5 symptom checklist (9 inattentive + 9
  hyperactive-impulsive): the clinical T-range 40–90 is scaled to a raw 0–36 sum
  and distributed across the nine items per subscale with sum-preserving jitter,
  so higher recorded scores yield higher answers. The original T-scores are kept
  inside the JSON for traceability.
- **Clinical narrative (`clinician_notes`)** — a free-text note per patient for
  the language model.

### 3.3 Label-leakage correction (notes regeneration)

The **first** notes generation stated the diagnosis verbatim (e.g. *"consistent
with ADHD – Combined Type"*), which leaked the label — an ablation showed the
notes alone scored a perfect (and meaningless) 1.000. The generator was rewritten
so the notes carry only **weak, realistic signal**: behaviours are mentioned
**probabilistically** (default **25 % signal / 75 % noise**, `--signal 0.25`), the
diagnosis is **never named**, ADHD and control notes draw on the **same
vocabulary** plus neutral non-diagnostic filler, and no severity words are tied to
scores. A TF-IDF probe of the corrected notes fell from perfect separation to
~0.61 (just above the majority baseline). See `LIMITATIONS.md` for the full arc.

### 3.4 Split and database ingestion

The generated files are split into the **seed set** (`*_1-129.csv`) and the
**held-out set** (`*_130-222.csv`, kept off the repo). `app/seed_db.py` loads the
seed set with the same cleaning/encryption helpers the API uses:

- phenotypic CSV → `Patient` (demographics; names encrypted) + `DSM5_Assessment`
  (DX label, Conners subscales, IQ, med status);
- DSM-5 CSV → `raw_answers` + `clinician_notes` on each assessment.

### 3.5 Feature extraction (serving-time contract)

`app/dsm5_features.py` is the shared train/serve feature builder: it embeds the
clinical note into a 768-d vector with a **frozen Bio_ClinicalBERT** (masked
mean-pooling) and concatenates it with 5 structured features (age, biological sex,
is-child, inattentive T-score, hyperactive T-score) → a **773-d** vector. Using the
identical function at train and inference time avoids train/serve skew.

---

## 4. Summary

| Stream | Raw source | Cleaning / synthesis | Prepared form | Label |
|--------|-----------|----------------------|---------------|-------|
| **MRI** | ADHD-200 Athena preprocessed (anatomical + functional) | `del` out `*_rest_*` / `*_csf*` / `*_wm*`; normalise 0–255; slice full axial stack → JPEG | 48,762 grayscale JPEG slices (2 scans × 189 × 129) in the `mri_images` volume, indexed by `MRI` rows | `ground_truth_dx` |
| **DSM-5 / NLP** | ADHD-200 NYU phenotypic table | synthetic names; T-score → 18-item questionnaire; 25 %-signal clinical notes | `Patient` + `DSM5_Assessment` rows → 773-d Bio_ClinicalBERT + demographic feature vector | `ground_truth_dx` |

Both datasets share the same de-identified cohort and the same diagnosis label,
so the image and text models are trained and evaluated on a consistent basis, and
their outputs can be combined into the aggregate ADHD risk score.
