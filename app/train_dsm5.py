"""
Anima - DSM-5 text & demographic model: training via stratified K-fold CV.

Trains the ADHD-risk head (dsm5_model.DSM5Head) on the frozen Bio_ClinicalBERT +
demographic features (dsm5_features). With only ~130 labelled patients we use
5-fold stratified cross-validation to make the most of every row: the model is
trained and tested 5 times on different splits, and the fold that generalises
best (highest test F1) is saved as the shipped artifact.

Pipeline (mirrors the request)
------------------------------
1. Connect to SQL Server via the existing .env / database.py.
2. Pull Patient + DSM5_Assessment rows that have a CONFIRMED diagnosis
   (ground_truth_dx IS NOT NULL) - controls (DX=0) included, since a binary
   ADHD-vs-control classifier needs both sides.
3. Turn every row into a (773,) feature vector with dsm5_features (the exact same
   function the API uses at inference - no train/serve skew).
4. Model = dsm5_model.DSM5Head: a single linear layer -> one logit -> sigmoid
   ADHD-risk. The same class the API loads. Kept tiny on purpose to learn signal
   rather than memorise; L2 weight decay + input standardisation regularise it.
5. StratifiedKFold(5). The hyperactive-only subtype (DX=2) is extremely rare, so
   it is merged into the other ADHD subtypes ONLY to form a valid stratification
   key (StratifiedKFold needs >= n_splits members per class); training still uses
   the binary label. Each fold's test-set subtype mix is printed so you can see
   the rare cases are spread across folds.
6. Standard PyTorch train/eval loop, run 5 times (once per fold).
7. Prints accuracy + F1 after each fold.
8. Tracks the highest-F1 fold.
9. Saves that winning fold's weights to app/models/dsm5_head.pt - exactly where
   DSM5_Analysis.py loads them, so the API picks the model up automatically.

XAI / explainability is intentionally NOT included yet.

Run (from the app/ folder, with the SQL Server container up and .env set):
    python train_dsm5.py
"""

import os
import sys
import copy
import argparse
import logging
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from database import get_connection
import dsm5_features
import dsm5_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("anima.train.dsm5")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(_HERE, "models", "dsm5_head.pt")

# DX code -> human label (ADHD-200 / NYU Athena convention).
DX_LABELS = {0: "Control", 1: "ADHD-Combined", 2: "ADHD-Hyper/Imp", 3: "ADHD-Inattentive"}
ADHD_DX = {1, 2, 3}


# =============================================================================
# 1 + 2. Data: pull the confirmed-diagnosis rows from SQL Server
# =============================================================================
def fetch_records() -> list:
    """Return one dict per patient with a confirmed diagnosis.

    Keys match dsm5_features.build_feature_matrix's expected inputs, plus the
    ground-truth DX used for the label and for stratification.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.patient_ID, p.age, p.biological_sex, p.is_child,
                   d.ground_truth_dx, d.inattentive_score, d.hyperactive_score,
                   d.clinician_notes
            FROM dbo.Patient AS p
            JOIN dbo.DSM5_Assessment AS d ON d.patient_ID = p.patient_ID
            WHERE d.ground_truth_dx IS NOT NULL
            ORDER BY p.patient_ID;
            """
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    records = []
    for (patient_id, age, biological_sex, is_child, dx,
         inattentive, hyperactive, notes) in rows:
        records.append({
            "patient_ID": patient_id,
            "age": age,
            "biological_sex": biological_sex,
            "is_child": is_child,
            "inattentive_score": inattentive,
            "hyperactive_score": hyperactive,
            "clinical_notes": notes,
            "dx": int(dx),
        })
    return records


# =============================================================================
# 5. Stratification key that is valid for K folds despite rare subtypes
# =============================================================================
def make_stratified_folds(dx: np.ndarray, n_splits: int, seed: int):
    """Build (train_idx, test_idx) folds with the rare subtype spread across folds.

    StratifiedKFold keeps each fold's subtype mix proportional, but it cannot
    guarantee a subtype with fewer than n_splits members lands in DIFFERENT test
    folds (with only 2 hyperactive cases it may dump both in one fold). So:

      * subtypes with >= n_splits members are split with sklearn StratifiedKFold
        (proportional, recognised, reproducible);
      * any rarer subtype (e.g. the 2 hyperactive cases) is allocated ROUND-ROBIN
        across folds, guaranteeing its members go to distinct test folds.

    Returns (folds, rare_classes) where folds is a list of (train_idx, test_idx).
    """
    dx = np.asarray(dx)
    N = len(dx)
    rng = np.random.default_rng(seed)
    counts = Counter(dx.tolist())
    rare = sorted(c for c in counts if counts[c] < n_splits)

    fold_of = np.full(N, -1, dtype=int)

    # Rare subtypes: round-robin (from a random start) -> spread over folds.
    for cls in rare:
        idx = np.where(dx == cls)[0]
        rng.shuffle(idx)
        start = int(rng.integers(n_splits))
        for j, i in enumerate(idx):
            fold_of[i] = (start + j) % n_splits
        logger.info(
            "  fold alloc: DX=%s (%s, n=%d) round-robin across folds "
            "(too rare for %d-way stratification).",
            cls, DX_LABELS.get(cls, cls), counts[cls], n_splits,
        )

    # Common subtypes: StratifiedKFold on the subtype label itself.
    common_idx = np.where(~np.isin(dx, rare))[0]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(common_idx, dx[common_idx])):
        fold_of[common_idx[te]] = f

    folds = [(np.where(fold_of != f)[0], np.where(fold_of == f)[0])
             for f in range(n_splits)]
    return folds, rare


def _subtype_breakdown(dx_subset: np.ndarray) -> str:
    counts = Counter(dx_subset.tolist())
    return "  ".join(f"{DX_LABELS.get(k, k)}={counts.get(k, 0)}"
                     for k in (0, 1, 3, 2))


# =============================================================================
# 6. Train / evaluate one fold
# =============================================================================
def train_one_fold(X_tr, y_tr, X_te, y_te, epochs, lr, weight_decay, note_pca_k=None):
    """Standard full-batch PyTorch loop; returns (model, accuracy, f1).

    If note_pca_k is set, a PCA is fitted on the TRAIN fold's note block and baked
    into the model so the 768-d embedding is projected to note_pca_k dims (the
    projection lives inside the model, so the API still feeds raw 773-d vectors).
    """
    model = dsm5_model.DSM5Head(input_dim=X_tr.shape[1], note_pca_k=note_pca_k)
    if note_pca_k:
        from sklearn.decomposition import PCA
        notes_tr = X_tr[:, dsm5_model.STRUCTURED_DIM:].cpu().numpy()
        pca = PCA(n_components=note_pca_k).fit(notes_tr)   # train-fold only (no leakage)
        model.set_pca(pca.mean_, pca.components_)
    # Standardise using TRAIN-fold stats of the PROCESSED features (no leakage).
    with torch.no_grad():
        proj_tr = model.project(X_tr)
    model.set_normalization(proj_tr.mean(dim=0), proj_tr.std(dim=0))

    # Class imbalance -> weight the positive (ADHD) class in the loss.
    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te))
    preds = (probs >= 0.5).int().cpu().numpy()
    y_true = y_te.int().cpu().numpy()
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    return model, acc, f1


def run_cv(X, y_t, dx, folds, epochs, lr, weight_decay, verbose=False, note_pca_k=None):
    """Run the CV loop over pre-built folds; returns per-fold metrics + best fold.

    Reused for the full model and each ablation / fusion config, always over the
    SAME folds so comparisons are apples-to-apples. Returns
    (accs, f1s, best) where best = {f1, fold, acc, state, config}.
    """
    accs, f1s = [], []
    best = {"f1": -1.0, "fold": None, "state": None, "acc": None, "config": None}
    for fold, (tr_idx, te_idx) in enumerate(folds, start=1):
        model, acc, f1 = train_one_fold(
            X[tr_idx], y_t[tr_idx], X[te_idx], y_t[te_idx],
            epochs=epochs, lr=lr, weight_decay=weight_decay, note_pca_k=note_pca_k,
        )
        accs.append(acc)
        f1s.append(f1)
        if verbose:
            logger.info(
                "Fold %d/%d | test n=%d | accuracy=%.3f | F1=%.3f | test subtypes: %s",
                fold, len(folds), len(te_idx), acc, f1, _subtype_breakdown(dx[te_idx]),
            )
        if f1 > best["f1"]:
            best.update(f1=f1, fold=fold, acc=acc,
                        state=copy.deepcopy(model.state_dict()), config=model.config())
    return accs, f1s, best


def _save_best(best, path) -> None:
    """Rebuild the winning fold's model from its config + weights and save it."""
    model = dsm5_model.DSM5Head(**best["config"])
    model.load_state_dict(best["state"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dsm5_model.save_head(model, path)


def run_experiment(X, y_t, dx, folds, args) -> int:
    """Compare fusion strategies (weight-decay sweep + note-PCA) on identical folds."""
    sd = dsm5_features.STRUCTURED_DIM
    # (label, feature matrix, note_pca_k, weight_decay)
    configs = [
        ("scores + demographics", X[:, :sd], None, args.weight_decay),
        ("notes only (BERT)", X[:, sd:], None, args.weight_decay),
    ]
    for wd in (0.01, 0.1, 0.5, 1.0):
        configs.append((f"full naive (wd={wd})", X, None, wd))
    for k in (8, 16, 32):
        configs.append((f"full PCA-{k} (wd={args.weight_decay})", X, k, args.weight_decay))

    logger.info("\n=== Fusion experiment | %d-fold CV, identical folds ===", args.folds)
    results = []
    best_full = {"f1": -1.0, "state": None, "config": None, "name": None, "acc": None}
    for name, X_cfg, k, wd in configs:
        accs, f1s, best = run_cv(X_cfg, y_t, dx, folds, args.epochs, args.lr, wd,
                                 note_pca_k=k)
        results.append((name, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)))
        logger.info("%-26s | accuracy=%.3f +/- %.3f | F1=%.3f +/- %.3f",
                    name, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s))
        if name.startswith("full") and np.mean(f1s) > best_full["f1"]:
            best_full.update(f1=np.mean(f1s), acc=np.mean(accs),
                             state=best["state"], config=best["config"], name=name)

    logger.info("\n=== Comparison (5-fold CV, ranked by F1) ===")
    logger.info("%-26s %-18s %-18s", "configuration", "accuracy", "F1")
    for name, am, asd, fm, fsd in sorted(results, key=lambda r: -r[3]):
        logger.info("%-26s %6.3f +/- %.3f    %6.3f +/- %.3f", name, am, asd, fm, fsd)

    logger.info("\nBest FULL configuration: %s (accuracy=%.3f, F1=%.3f)",
                best_full["name"], best_full["acc"], best_full["f1"])
    _save_best(best_full, args.output)
    logger.info("Saved best full-model weights -> %s", args.output)
    logger.info("DSM5_Analysis.py will load this automatically on the next call.")
    return 0


# =============================================================================
# Entry point
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Train the DSM-5 ADHD-risk head with K-fold CV.")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds (>=5).")
    ap.add_argument("--epochs", type=int, default=300, help="Training epochs per fold.")
    ap.add_argument("--lr", type=float, default=0.01, help="Adam learning rate.")
    ap.add_argument("--weight-decay", type=float, default=1e-2,
                    help="L2 regularisation (higher = simpler model).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (reproducibility).")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Where to save the winning fold's weights.")
    ap.add_argument("--no-ablation", dest="ablation", action="store_false",
                    help="Skip the scores-only / notes-only ablation comparison.")
    ap.add_argument("--note-pca", type=int, default=None,
                    help="Project the 768-d note embedding to this many PCA dims "
                         "before fusion (mitigates the notes swamping the scores).")
    ap.add_argument("--experiment", action="store_true",
                    help="Run the fusion sweep (weight-decay + note-PCA) and rank them.")
    ap.set_defaults(ablation=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 1 + 2. Load labelled data --------------------------------------------
    logger.info("Loading confirmed-diagnosis records from SQL Server ...")
    records = fetch_records()
    n = len(records)
    if n < args.folds:
        sys.exit(f"ERROR: only {n} labelled patients - need at least {args.folds}.")

    dx = np.array([r["dx"] for r in records])
    y = np.array([1 if d in ADHD_DX else 0 for d in dx])

    logger.info("Loaded %d patients (confirmed diagnosis).", n)
    logger.info("  Binary:   ADHD=%d  Control=%d", int((y == 1).sum()), int((y == 0).sum()))
    logger.info("  Subtypes: %s", _subtype_breakdown(dx))

    # ---- 3. Features (computed ONCE for all patients, then reused per fold) ----
    logger.info("Building features via Bio_ClinicalBERT (first run downloads the model) ...")
    X = dsm5_features.build_feature_matrix(records).float()
    y_t = torch.tensor(y, dtype=torch.float32)
    logger.info("Feature matrix: %s", tuple(X.shape))

    # ---- 5. Stratified folds (rare hyperactive spread across folds) -----------
    # Built ONCE and reused for the full model and every ablation slice, so all
    # configurations are compared on identical train/test partitions.
    folds, _ = make_stratified_folds(dx, args.folds, args.seed)

    # ---- Optional: full fusion sweep (weight-decay + note-PCA) -----------------
    if args.experiment:
        return run_experiment(X, y_t, dx, folds, args)

    # ---- 6 + 7 + 8. FULL model: train/test 5 times, report, track the best ----
    pca_note = f" | note-PCA={args.note_pca}" if args.note_pca else ""
    logger.info("\n=== FULL model | %d-fold stratified cross-validation ===", args.folds)
    logger.info("features: scores + demographics + clinical-note embedding (%d dims)%s",
                X.shape[1], pca_note)
    full_accs, full_f1s, best = run_cv(
        X, y_t, dx, folds, args.epochs, args.lr, args.weight_decay, verbose=True,
        note_pca_k=args.note_pca,
    )
    logger.info("Summary | accuracy=%.3f +/- %.3f | F1=%.3f +/- %.3f | best fold #%d",
                np.mean(full_accs), np.std(full_accs),
                np.mean(full_f1s), np.std(full_f1s), best["fold"])

    # ---- Ablation: where does the signal actually come from? ------------------
    # Same folds, same loop; only the feature columns fed to the model change.
    #   structured = age, sex, is_child, inattentive + hyperactive T-scores (5)
    #   notes      = the 768-d Bio_ClinicalBERT embedding of the clinical note
    results = [("full (scores + demographics + notes)", full_accs, full_f1s)]
    if args.ablation:
        sd = dsm5_features.STRUCTURED_DIM
        slices = [
            ("scores + demographics only", X[:, :sd]),
            ("clinical notes only (BERT)", X[:, sd:]),
        ]
        logger.info("\n=== Ablation | same folds, different feature subsets ===")
        for name, X_sub in slices:
            accs, f1s, _ = run_cv(
                X_sub, y_t, dx, folds, args.epochs, args.lr, args.weight_decay,
            )
            results.append((name, accs, f1s))
            logger.info("%-34s | accuracy=%.3f +/- %.3f | F1=%.3f +/- %.3f",
                        name, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s))

        # Side-by-side comparison table.
        logger.info("\n=== Feature-set comparison (5-fold CV) ===")
        logger.info("%-38s %-16s %-16s", "feature set", "accuracy", "F1")
        for name, accs, f1s in results:
            logger.info("%-38s %6.3f +/- %.3f   %6.3f +/- %.3f",
                        name, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s))
        logger.info("Read this as: if 'notes only' rivals 'full', the synthetic "
                    "notes are leaking the label; if 'full' <= 'scores only', the "
                    "text adds no independent signal on this dataset.")

    # ---- 9. Save the winning FULL-model weights (what the API loads) ----------
    _save_best(best, args.output)
    logger.info("\nSaved winning FULL model weights -> %s", args.output)
    logger.info("DSM5_Analysis.py will load this automatically on the next call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
