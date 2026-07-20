"""
Anima - Combined Risk Analysis READ-side live validation.

Exercises the RBAC-gated retrieval endpoints against the real database, without
needing the API server running (it calls the endpoint functions directly with a
live connection):

  * GET latest result      -> Combined_Analysis.get_latest_result
  * GET full history        -> Combined_Analysis.get_result_history
  * RBAC negative check      -> an unauthorised requester must be denied

Run (from app/, DB up + seeded with at least one Analysis_Result row, .env set):
    python combined_read_smoketest.py --patient 0010001
    python combined_read_smoketest.py --patient 0010001 --as A000001
"""

import sys
import argparse

from fastapi import HTTPException

from database import get_connection
import Combined_Analysis as C


def main() -> int:
    ap = argparse.ArgumentParser(description="Live read-side validation for Analysis_Result.")
    ap.add_argument("--patient", default="0010001", help="patient_ID to read.")
    ap.add_argument("--as", dest="as_user", default="A000001",
                    help="requester user_ID (default: seeded admin A000001).")
    args = ap.parse_args()
    pid = args.patient.strip().split(".")[0].zfill(7)

    # ---- 1. latest + history as an authorised requester ----------------------
    print(f"Reading analysis results for patient {pid} as {args.as_user} ...\n")
    conn = get_connection()
    try:
        try:
            latest = C.get_latest_result(pid, args.as_user, conn=conn)
            r = latest["result"]
            print(f"  latest  -> analysis #{r['analysis_ID']} | combined="
                  f"{r['final_combined_score']} | subtype={r['predicted_subtype']} | "
                  f"nlp={r['nlp_risk_score']} mri={r['mri_risk_score']} | "
                  f"created_by={r['created_by']} | at {r['created_at']}")
        except HTTPException as exc:
            print(f"  latest  -> HTTP {exc.status_code}: {exc.detail}")

        hist = C.get_result_history(pid, args.as_user, conn=conn)
        print(f"  history -> {hist['count']} row(s):")
        for h in hist["history"]:
            print(f"     #{h['analysis_ID']}  combined={h['final_combined_score']}  "
                  f"nlp={h['nlp_risk_score']}  mri={h['mri_risk_score']}  "
                  f"subtype={h['predicted_subtype']}  created_by={h['created_by']}  "
                  f"at {h['created_at']}")
    finally:
        conn.close()

    # ---- 2. RBAC negative check: an unauthorised requester is denied ----------
    print("\nRBAC check: an unauthorised requester must be denied ...")
    conn = get_connection()
    try:
        C.get_latest_result(pid, "C9999999", conn=conn)   # unknown/unassigned
        print("  WARNING: expected a denial (401/403) but a result was returned!")
        rc = 1
    except HTTPException as exc:
        if exc.status_code in (401, 403):
            print(f"  OK: denied with HTTP {exc.status_code} ({exc.detail})")
            rc = 0
        else:
            print(f"  Unexpected HTTP {exc.status_code}: {exc.detail}")
            rc = 1
    finally:
        conn.close()

    print("\nRead-side validation complete.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
