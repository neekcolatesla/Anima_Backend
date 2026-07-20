"""
Anima - Combined Risk Analysis live smoke test.

Runs the REAL `Combined_Analysis.analyze_combined` against the seeded database for
one or more patients and reads back the `Analysis_Result` audit row it wrote, so
you can confirm the whole chain end-to-end: DSM-5 scoring -> MRI scoring (or
graceful fallback) -> fusion -> append-only audit persistence.

Run (from app/, DB up + seeded, .env set):
    python combined_smoketest.py --patient 0010001
    python combined_smoketest.py --limit 5          # first 5 labelled patients
    python combined_smoketest.py --as A000001       # who is triggering (created_by)
"""

import sys
import argparse

from database import get_connection
from Combined_Analysis import analyze_combined


def _fetch_patients(limit, patient):
    """Pick target patient IDs: an explicit one, or the first N labelled patients."""
    if patient:
        return [patient.strip().split(".")[0].zfill(7)]
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TOP (?) patient_ID FROM dbo.DSM5_Assessment "
            "WHERE ground_truth_dx IS NOT NULL ORDER BY patient_ID;", limit)
        return [str(r[0]) for r in cursor.fetchall()]
    finally:
        conn.close()


def _latest_audit(patient_id):
    """Read back the newest Analysis_Result row for a patient (proves the write)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TOP 1 analysis_ID, nlp_risk_score, mri_risk_score, "
            "final_combined_score, predicted_subtype, model_version, created_by, "
            "created_at FROM dbo.Analysis_Result WHERE patient_ID = ? "
            "ORDER BY analysis_ID DESC;", patient_id)
        return cursor.fetchone()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live smoke test for Combined Risk Analysis.")
    ap.add_argument("--patient", default=None, help="A single patient_ID to analyse.")
    ap.add_argument("--limit", type=int, default=3,
                    help="Analyse the first N labelled patients (if --patient omitted).")
    ap.add_argument("--as", dest="as_user", default="A000001",
                    help="user_ID recorded as created_by (default: seeded admin A000001).")
    args = ap.parse_args()

    patients = _fetch_patients(args.limit, args.patient)
    if not patients:
        sys.exit("No patients found - seed the database first (seed_dsm5.py).")

    print(f"Running Combined Risk Analysis on {len(patients)} patient(s) "
          f"as {args.as_user}...\n")
    ok = 0
    for pid in patients:
        try:
            res = analyze_combined(pid, created_by=args.as_user)
        except Exception as exc:
            print(f"  FAIL {pid}: {exc}")
            continue

        print(f"  {pid}: combined={res['final_combined_score']} "
              f"({res['predicted_diagnosis']}) | nlp={res['nlp_risk_score']} "
              f"mri={res['mri_risk_score']} ({res['mri_status']}) | "
              f"subtype={res['predicted_subtype']} | weights[{res['weighting']}]")

        audit = _latest_audit(pid)
        if audit:
            print(f"       -> audit row #{audit[0]} persisted: "
                  f"combined={audit[3]} subtype={audit[4]} "
                  f"model={audit[5]} created_by={audit[6]} at {audit[7]}")
            ok += 1
        else:
            print("       -> WARNING: no Analysis_Result row found (persistence failed)!")

    print(f"\nDone. {ok}/{len(patients)} patient(s) produced a persisted audit row.")
    return 0 if ok == len(patients) else 1


if __name__ == "__main__":
    sys.exit(main())
