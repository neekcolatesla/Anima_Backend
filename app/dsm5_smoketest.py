"""
Anima - DSM-5 analysis smoke test (seeded 1-129 patients).

Exercises the REAL inference path (DSM5_Analysis.analyze_patient) end-to-end
against the patients already in SQL Server: loads the trained head, builds the
Bio_ClinicalBERT + demographic features, scores an ADHD-risk value, predicts the
subtype, and writes nlp_risk_score back to the DB - exactly what the
POST /api/analysis/dsm5/{id} endpoint does.

SCOPE / CAVEAT
--------------
This is a FUNCTIONAL smoke test, not a performance evaluation. The model was
trained and cross-validated on these same 1-129 patients, so scoring them again
is optimistic by construction - the honest metrics are the 5-fold CV numbers in
LIMITATIONS.md. This script answers "does the trained model load and produce
sensible, non-crashing outputs through the real endpoint code?". The held-out
130-222 block is never touched (it was never seeded).

Run (from app/, DB up, .env set, trained head present):
    python dsm5_smoketest.py            # all seeded patients
    python dsm5_smoketest.py --limit 15 # first 15 only (faster)
    python dsm5_smoketest.py --patient 0010001
"""

import os
import sys
import argparse

from database import get_connection
import dsm5_model
from DSM5_Analysis import analyze_patient, HEAD_PATH

DX_LABELS = {0: "Control", 1: "ADHD-Combined", 2: "ADHD-Hyper/Imp", 3: "ADHD-Inattentive"}
ADHD_DX = {1, 2, 3}


def _fetch_seeded():
    """Return [(patient_ID, ground_truth_dx)] for confirmed-diagnosis patients."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT p.patient_ID, d.ground_truth_dx "
            "FROM dbo.Patient AS p "
            "JOIN dbo.DSM5_Assessment AS d ON d.patient_ID = p.patient_ID "
            "WHERE d.ground_truth_dx IS NOT NULL "
            "ORDER BY p.patient_ID;"
        )
        return [(str(pid).strip(), int(dx)) for pid, dx in cur.fetchall()]
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test the DSM-5 analysis endpoint.")
    ap.add_argument("--limit", type=int, default=None, help="Only score the first N patients.")
    ap.add_argument("--patient", default=None, help="Score a single patient_ID and exit.")
    args = ap.parse_args()

    # 1. Report which model the endpoint will use.
    if os.path.exists(HEAD_PATH):
        cfg = dsm5_model.load_head(HEAD_PATH).config()
        kind = f"PCA-{cfg['note_pca_k']} fusion" if cfg["note_pca_k"] else "plain linear"
        print(f"Trained head found: {HEAD_PATH}")
        print(f"  architecture: {kind}  config={cfg}")
    else:
        print(f"WARNING: no trained head at {HEAD_PATH} - endpoint will use the "
              f"heuristic fallback. Run train_dsm5.py first.")

    # 2. Single-patient mode.
    if args.patient:
        print(f"\n{analyze_patient(args.patient.strip())}")
        return 0

    patients = _fetch_seeded()
    if not patients:
        sys.exit("No seeded patients with a confirmed diagnosis - run seed_dsm5.py first.")
    if args.limit:
        patients = patients[: args.limit]

    print(f"\nScoring {len(patients)} seeded patient(s) through analyze_patient() ...\n")
    header = f"{'patient_ID':<12}{'true DX':<18}{'risk':>7}  {'predicted subtype':<18}{'method':<10}"
    print(header)
    print("-" * len(header))

    results = []
    for pid, dx in patients:
        try:
            r = analyze_patient(pid)
        except Exception as exc:                       # keep going on a bad row
            print(f"{pid:<12}ERROR: {exc}")
            continue
        r["dx"] = dx
        results.append(r)
        print(f"{pid:<12}{DX_LABELS.get(dx, dx):<18}{r['nlp_risk_score']:>7.2f}  "
              f"{str(r['predicted_subtype']):<18}{r['scoring_method']:<10}")

    # 3. Summary / sanity checks.
    if not results:
        sys.exit("No patients scored successfully.")
    n = len(results)
    n_model = sum(1 for r in results if r["scoring_method"] == "model")
    adhd = [r for r in results if r["dx"] in ADHD_DX]
    ctrl = [r for r in results if r["dx"] == 0]
    mean = lambda rs: (sum(r["nlp_risk_score"] for r in rs) / len(rs)) if rs else float("nan")

    # Binary agreement on the training data (optimistic - smoke check only).
    correct = sum(1 for r in results
                  if (r["nlp_risk_score"] >= 50) == (r["dx"] in ADHD_DX))

    fmt = lambda rs: f"{mean(rs):.2f}" if rs else "n/a"
    print("\n=== Summary (functional smoke test - NOT an evaluation) ===")
    print(f"Scored:                 {n}")
    print(f"Used trained model:     {n_model}/{n}  (rest fell back to heuristic)")
    print(f"Mean risk - ADHD:       {fmt(adhd)}   (n={len(adhd)})")
    print(f"Mean risk - Control:    {fmt(ctrl)}   (n={len(ctrl)})")
    print(f"Risk>=50 matches label: {correct}/{n} ({100.0*correct/n:.1f}%)  "
          f"[optimistic: model saw these patients in training]")

    subtypes = {}
    for r in results:
        subtypes[r["predicted_subtype"]] = subtypes.get(r["predicted_subtype"], 0) + 1
    print("Predicted subtypes:     " + ", ".join(f"{k}={v}" for k, v in sorted(
        subtypes.items(), key=lambda kv: str(kv[0]))))

    if n_model < n:
        print("\nNOTE: some rows used the heuristic fallback - check the trained head "
              "loads (torch/transformers installed, head file present).")
    if not (adhd and ctrl):
        print("\nNOTE: sample lacked both classes (use no --limit for the "
              "ADHD-vs-control sanity check).")
    elif mean(adhd) <= mean(ctrl):
        print("\nWARNING: mean ADHD risk is not above control risk - inspect the model.")
    else:
        print("\nOK: ADHD risk > control risk, trained model in use, endpoint path works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
