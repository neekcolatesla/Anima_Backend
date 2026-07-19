"""
Anima - MRI image classification model: training via patient-level K-fold CV.

Trains the shallow CNN (mri_model.MRICNN) to classify a patient's anatomical MRI
slices as ADHD vs control, and saves the best fold's weights where the serving
engine (MRI_Analysis.py) loads them. Mirrors train_dsm5.py in spirit: 5-fold
stratified cross-validation over a tiny cohort, per-fold accuracy printed, the
best-generalising fold shipped.

Pipeline (mirrors the request)
------------------------------
1. Connect to SQL Server via the existing .env / database.py and pull each
   patient's TRUE label from DSM5_Assessment.ground_truth_dx (binary: ADHD vs
   control - controls, DX=0, included, since the classifier needs both sides).
2. For each patient, locate their CURRENT image folders in the MRI table
   (is_current = 1): the structural-anatomy stack (scan_type='anat') and the
   smoothed grey-matter stack (scan_type='anat_gm').
3. Pair the two stacks positionally, resize to input_size, and stack them into a
   2-channel sample (channel 0 = anat, channel 1 = anat_gm). Done lazily by a
   Dataset so we never hold the whole ~48k-slice set in RAM.
4. 5-fold stratified CV split BY PATIENT ID, never by slice. Every slice from a
   test patient is in the test set - so no slice from a brain ever appears in
   both train and test (no data leakage across the split).
5. Import MRICNN from mri_model and train ONLY that shallow architecture, so it
   learns real anatomical differences instead of memorising ~130 patients.
6. The network outputs one logit (one ADHD prediction) PER SLICE; it is trained
   on slices with the BCE loss, each slice carrying its patient's label.
7. After each fold, print the test accuracy (slice-level and patient-level).
8. Track which fold generalises best on its held-out test patients.
9. On finish, save the winning fold's weights with mri_model.save_cnn to
   app/models/mri_cnn.pt - exactly where MRI_Analysis.py loads them, so the API
   picks the model up automatically (no trained-model "pending" any more).

Patient-level vs slice-level (why two accuracies)
-------------------------------------------------
The CNN is a per-slice classifier, but at serving MRI_Analysis.py aggregates a
patient's slices (mean probability) into ONE decision. So the honest "test
performance" is patient-level, and the best fold is chosen on patient-level
accuracy (tie-broken by F1). Slice-level accuracy is printed too, as a lower-level
health check.

XAI / explainability is intentionally NOT included yet.

Run (from the app/ folder, with SQL Server up + patients/MRI seeded, .env set):
    python train_mri.py
    python train_mri.py --epochs 15 --max-slices 60     # faster / lighter
"""

import os
import sys
import copy
import argparse
import logging
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from database import get_connection
import mri_model
import mri_slices

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("anima.train.mri")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(_HERE, "models", "mri_cnn.pt")

# DX code -> human label (ADHD-200 / NYU Athena convention).
DX_LABELS = {0: "Control", 1: "ADHD-Combined", 2: "ADHD-Hyper/Imp", 3: "ADHD-Inattentive"}
ADHD_DX = {1, 2, 3}


# =============================================================================
# 1 + 2. Data: labels from DSM5_Assessment, current scan folders from MRI
# =============================================================================
def fetch_patient_scans() -> list:
    """Return one dict per usable patient: {patient_ID, dx, anat_dir, anat_gm_dir}.

    A patient is usable only if they have a confirmed ground_truth_dx AND both
    current (is_current=1) scan folders. Others are skipped with a warning.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Labels: one confirmed diagnosis per patient.
        cursor.execute(
            "SELECT patient_ID, ground_truth_dx FROM dbo.DSM5_Assessment "
            "WHERE ground_truth_dx IS NOT NULL;"
        )
        labels = {}
        for patient_id, dx in cursor.fetchall():
            labels[str(patient_id)] = int(dx)

        # Current scan folders per patient.
        cursor.execute(
            "SELECT patient_ID, scan_type, file_path FROM dbo.MRI "
            "WHERE is_current = 1 AND scan_type IN ('anat', 'anat_gm');"
        )
        scans = defaultdict(dict)
        for patient_id, scan_type, file_path in cursor.fetchall():
            scans[str(patient_id)][scan_type] = file_path
    finally:
        conn.close()

    records, skipped = [], 0
    for patient_id in sorted(scans):
        both = scans[patient_id]
        if "anat" not in both or "anat_gm" not in both:
            skipped += 1
            continue
        if patient_id not in labels:
            skipped += 1
            continue
        records.append({
            "patient_ID": patient_id,
            "dx": labels[patient_id],
            "anat_dir": both["anat"],
            "anat_gm_dir": both["anat_gm"],
        })
    if skipped:
        logger.info("  (skipped %d patient(s) missing a label or a scan folder)", skipped)
    return records


def build_sample_index(records: list, central_frac, min_foreground, max_slices) -> tuple:
    """Turn patient records into a flat list of per-slice training samples.

    Returns (samples, patient_labels) where:
      * samples = list of (anat_path, gm_path, label, patient_ID) - one per PAIRED
        slice; anat[i] is paired with anat_gm[i] (both are the same ordered axial
        series), and label is the patient's binary ADHD/control target;
      * patient_labels = {patient_ID: label}.

    Slice SELECTION (central crop -> optional foreground filter -> optional even
    subsample) is delegated to the shared mri_slices.pair_slices so the trainer
    and the serving engine (MRI_Analysis.py) pick the SAME slices - no skew.
    """
    samples = []
    patient_labels = {}
    for rec in records:
        label = 1 if rec["dx"] in ADHD_DX else 0
        patient_labels[rec["patient_ID"]] = label

        pairs = mri_slices.pair_slices(
            rec["anat_dir"], rec["anat_gm_dir"],
            central_frac=central_frac, min_foreground=min_foreground,
            max_slices=max_slices,
        )
        if not pairs:
            logger.warning("  patient %s has no readable slices; skipping.",
                           rec["patient_ID"])
            patient_labels.pop(rec["patient_ID"], None)
            continue
        for a_path, g_path in pairs:
            samples.append((a_path, g_path, label, rec["patient_ID"]))
    return samples, patient_labels


# =============================================================================
# 3. Dataset: pair -> resize -> 2-channel tensor (lazy, from disk)
# =============================================================================
class SliceDataset(Dataset):
    """A paired-slice dataset. Each item is a (2, H, W) tensor + label + patient."""

    def __init__(self, samples: list, input_size: int):
        self.samples = samples
        self.input_size = int(input_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L").resize((self.input_size, self.input_size))
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, i: int):
        a_path, g_path, label, patient_id = self.samples[i]
        pair = np.stack([self._load(a_path), self._load(g_path)], axis=0)  # (2,H,W)
        x = torch.from_numpy(pair)
        return x, torch.tensor(float(label)), patient_id


# =============================================================================
# 4. Patient-level stratified folds (NO slice leakage)
# =============================================================================
def make_patient_folds(patient_ids: list, labels: list, n_splits: int, seed: int):
    """StratifiedKFold over PATIENTS. Returns list of (train_pids, test_pids) sets.

    Splitting by patient (not by slice) is the guarantee that all of a brain's
    slices land on the same side of every split.
    """
    pids = np.array(patient_ids)
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(pids, y):
        folds.append((set(pids[tr].tolist()), set(pids[te].tolist())))
    return folds


def _label_breakdown(labels) -> str:
    counts = Counter(int(v) for v in labels)
    return f"ADHD={counts.get(1, 0)}  Control={counts.get(0, 0)}"


# =============================================================================
# 5 + 6. Train / evaluate one fold
# =============================================================================
def _make_loader(samples, input_size, batch_size, shuffle, num_workers, device):
    return DataLoader(
        SliceDataset(samples, input_size),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )


def train_one_fold(train_samples, test_samples, patient_labels, args, device):
    """Train MRICNN on the fold's TRAIN slices, evaluate on its TEST slices.

    Returns (model, metrics) where metrics = {slice_acc, patient_acc, patient_f1}.
    The model outputs one logit per slice; patient-level metrics aggregate a
    patient's slice probabilities by mean (exactly as MRI_Analysis.py serves).
    """
    model = mri_model.MRICNN(input_size=args.input_size, dropout=args.dropout).to(device)

    # Class imbalance -> weight the positive (ADHD) class in the loss.
    tr_labels = [s[2] for s in train_samples]
    n_pos = float(sum(1 for l in tr_labels if l == 1))
    n_neg = float(sum(1 for l in tr_labels if l == 0))
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_loader = _make_loader(train_samples, args.input_size, args.batch_size,
                                True, args.num_workers, device)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for x, y, _pid in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)                     # (N,) per-slice logits
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        if args.verbose_epochs:
            logger.info("    epoch %2d/%d | train loss=%.4f",
                        epoch + 1, args.epochs, running / max(len(train_samples), 1))

    # ---- evaluate on the held-out test patients ----
    test_loader = _make_loader(test_samples, args.input_size, args.batch_size,
                               False, args.num_workers, device)
    model.eval()
    slice_true, slice_prob, slice_pid = [], [], []
    with torch.no_grad():
        for x, y, pid in test_loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            slice_prob.extend(probs.tolist())
            slice_true.extend(y.numpy().tolist())
            slice_pid.extend(list(pid))

    slice_prob = np.array(slice_prob)
    slice_true = np.array(slice_true).astype(int)
    slice_pred = (slice_prob >= 0.5).astype(int)
    slice_acc = accuracy_score(slice_true, slice_pred) if len(slice_true) else 0.0

    # Aggregate to one decision per patient (mean prob -> threshold), as served.
    by_patient = defaultdict(list)
    for p, prob in zip(slice_pid, slice_prob):
        by_patient[p].append(prob)
    p_true, p_pred = [], []
    for pid, probs in by_patient.items():
        p_true.append(patient_labels[pid])
        p_pred.append(1 if float(np.mean(probs)) >= args.threshold else 0)
    patient_acc = accuracy_score(p_true, p_pred) if p_true else 0.0
    patient_f1 = f1_score(p_true, p_pred, zero_division=0) if p_true else 0.0

    return model, {"slice_acc": slice_acc, "patient_acc": patient_acc,
                   "patient_f1": patient_f1, "n_test_patients": len(p_true)}


# =============================================================================
# 7 + 8. Cross-validation loop
# =============================================================================
def run_cv(samples, patient_labels, folds, args, device):
    """Run K-fold CV; print per-fold accuracy; return per-fold metrics + best fold."""
    by_patient_samples = defaultdict(list)
    for s in samples:
        by_patient_samples[s[3]].append(s)

    slice_accs, patient_accs, patient_f1s = [], [], []
    best = {"patient_acc": -1.0, "patient_f1": -1.0, "fold": None,
            "state": None, "config": None}

    for fold, (train_pids, test_pids) in enumerate(folds, start=1):
        # Safety net: the split must never share a patient across train/test.
        assert not (train_pids & test_pids), "patient leaked across the split!"

        train_samples = [s for p in train_pids for s in by_patient_samples[p]]
        test_samples = [s for p in test_pids for s in by_patient_samples[p]]

        model, m = train_one_fold(train_samples, test_samples, patient_labels,
                                  args, device)
        slice_accs.append(m["slice_acc"])
        patient_accs.append(m["patient_acc"])
        patient_f1s.append(m["patient_f1"])

        logger.info(
            "Fold %d/%d | train patients=%d slices=%d | test patients=%d slices=%d | "
            "patient acc=%.3f  F1=%.3f | slice acc=%.3f",
            fold, len(folds), len(train_pids), len(train_samples),
            m["n_test_patients"], len(test_samples),
            m["patient_acc"], m["patient_f1"], m["slice_acc"],
        )

        # Best fold = best patient-level accuracy (deployment metric), tie -> F1.
        better = (m["patient_acc"] > best["patient_acc"] or
                  (m["patient_acc"] == best["patient_acc"] and
                   m["patient_f1"] > best["patient_f1"]))
        if better:
            best.update(patient_acc=m["patient_acc"], patient_f1=m["patient_f1"],
                        fold=fold, state=copy.deepcopy(model.state_dict()),
                        config=model.config())

    return slice_accs, patient_accs, patient_f1s, best


def _save_best(best, path) -> None:
    """Rebuild the winning fold's MRICNN from its config + weights and save it."""
    model = mri_model.MRICNN(**best["config"])
    model.load_state_dict(best["state"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mri_model.save_cnn(model, path)


# =============================================================================
# Optional: small hyperparameter sweep (exploratory)
# =============================================================================
# Curated grid of (label, overrides) run over the SAME folds so comparisons are
# apples-to-apples. Deliberately small - each entry is a full K-fold CV run.
SWEEP_GRID = [
    ("baseline",        {}),
    ("more epochs",     {"epochs": 25}),
    ("lower lr",        {"lr": 3e-4}),
    ("stronger reg",    {"weight_decay": 1e-3, "dropout": 0.5}),
    ("bigger input",    {"input_size": 160}),
]


def run_sweep(samples, patient_labels, folds, base_args, device) -> int:
    """Run the curated grid on identical folds, rank by patient accuracy, save best.

    EXPLORATORY. Because the grid is chosen by looking at the same held-out folds
    it is scored on, treat any 'win' cautiously - a real gain should survive a
    fresh seed / fold split, not just top this table. Structural-MRI ADHD is a
    hard problem (see README_MRI.md); do not read a small bump as a solved model.
    """
    logger.info("\n=== Hyperparameter sweep | %d-fold CV, identical folds ===",
                base_args.folds)
    logger.info("(exploratory - see the caveat in the code / README_MRI.md)\n")

    results = []
    best_overall = {"patient_acc": -1.0, "patient_f1": -1.0,
                    "state": None, "config": None, "name": None}
    for name, overrides in SWEEP_GRID:
        args = copy.copy(base_args)
        for k, v in overrides.items():
            setattr(args, k, v)
        _, patient_accs, patient_f1s, best = run_cv(
            samples, patient_labels, folds, args, device)
        pa, pasd = float(np.mean(patient_accs)), float(np.std(patient_accs))
        fa, fasd = float(np.mean(patient_f1s)), float(np.std(patient_f1s))
        results.append((name, pa, pasd, fa, fasd, overrides))
        logger.info("%-14s | patient acc=%.3f +/- %.3f | F1=%.3f +/- %.3f | %s",
                    name, pa, pasd, fa, fasd, overrides or "(defaults)")
        if best["patient_acc"] > best_overall["patient_acc"]:
            best_overall.update(patient_acc=best["patient_acc"],
                                patient_f1=best["patient_f1"],
                                state=best["state"], config=best["config"], name=name)

    logger.info("\n=== Sweep ranking (by mean patient accuracy) ===")
    for name, pa, pasd, fa, fasd, ov in sorted(results, key=lambda r: -r[1]):
        logger.info("%-14s  acc=%.3f +/- %.3f   F1=%.3f +/- %.3f", name, pa, pasd, fa, fasd)

    logger.info("\nBest config: %s (best-fold patient acc=%.3f)",
                best_overall["name"], best_overall["patient_acc"])
    _save_best(best_overall, base_args.output)
    logger.info("Saved best sweep weights -> %s", base_args.output)
    return 0


# =============================================================================
# Entry point
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Train the MRI ADHD CNN with patient-level K-fold CV.")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds (>=5).")
    ap.add_argument("--epochs", type=int, default=12, help="Training epochs per fold.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 regularisation (higher = simpler model).")
    ap.add_argument("--batch-size", type=int, default=64, help="Slices per batch.")
    ap.add_argument("--input-size", type=int, default=mri_model.INPUT_SIZE,
                    help="Square edge each slice is resized to.")
    ap.add_argument("--dropout", type=float, default=0.4, help="CNN dropout rate.")
    ap.add_argument("--central-frac", type=float, default=0.6,
                    help="Keep the middle fraction of the axial stack (skip empty "
                         "edge slices). 1.0 = use the whole stack.")
    ap.add_argument("--min-foreground", type=float, default=0.0,
                    help="Drop slices whose fraction of brain-tissue pixels is below "
                         "this (0 = keep all central slices).")
    ap.add_argument("--max-slices", type=int, default=0,
                    help="Cap paired slices per patient (0 = all; evenly sampled).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Mean-probability cut-off for the patient-level ADHD call.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (0 is safest on Windows).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (reproducibility).")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Where to save the winning fold's weights.")
    ap.add_argument("--verbose-epochs", action="store_true",
                    help="Print per-epoch training loss.")
    ap.add_argument("--sweep", action="store_true",
                    help="Run the exploratory hyperparameter grid and rank configs.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- 1 + 2. Load labels + current scan folders ----------------------------
    logger.info("Loading labelled patients with current MRI scans from SQL Server ...")
    records = fetch_patient_scans()
    if len(records) < args.folds:
        sys.exit(f"ERROR: only {len(records)} usable patients - need at least {args.folds}.")

    # ---- 3. Flatten to per-slice samples (shared selection: no train/serve skew)
    samples, patient_labels = build_sample_index(
        records, args.central_frac, args.min_foreground, args.max_slices)
    pids = sorted(patient_labels)
    labels = [patient_labels[p] for p in pids]
    logger.info("Slice selection: central_frac=%.2f  min_foreground=%.2f  max_slices=%s",
                args.central_frac, args.min_foreground, args.max_slices or "all")
    logger.info("Usable patients: %d (%s)", len(pids), _label_breakdown(labels))
    logger.info("Total paired slices: %d (avg %.0f per patient)",
                len(samples), len(samples) / max(len(pids), 1))

    if len(set(labels)) < 2:
        sys.exit("ERROR: need both ADHD and control patients to train a classifier.")

    # ---- 4. Patient-level stratified folds ------------------------------------
    folds = make_patient_folds(pids, labels, args.folds, args.seed)

    # ---- Optional: exploratory hyperparameter sweep ---------------------------
    if args.sweep:
        return run_sweep(samples, patient_labels, folds, args, device)

    # ---- 5-8. Cross-validated training ----------------------------------------
    logger.info("\n=== MRICNN | %d-fold stratified CV (split BY PATIENT) ===", args.folds)
    slice_accs, patient_accs, patient_f1s, best = run_cv(
        samples, patient_labels, folds, args, device)

    logger.info(
        "\nSummary | patient acc=%.3f +/- %.3f | patient F1=%.3f +/- %.3f | "
        "slice acc=%.3f +/- %.3f | best fold #%d (patient acc=%.3f)",
        np.mean(patient_accs), np.std(patient_accs),
        np.mean(patient_f1s), np.std(patient_f1s),
        np.mean(slice_accs), np.std(slice_accs),
        best["fold"], best["patient_acc"],
    )

    # ---- 9. Save the winning fold's weights (what the API loads) --------------
    _save_best(best, args.output)
    logger.info("\nSaved winning MRICNN weights -> %s", args.output)
    logger.info("MRI_Analysis.py will load this automatically on the next call "
                "(status flips from 'pending' to 'success').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
