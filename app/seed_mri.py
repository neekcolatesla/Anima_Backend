"""
Anima - MRI training seeder (dev / evaluation helper).

Bulk-ingests the training patients' MRI scans from a directory of per-patient
folders, e.g.

    <mri-dir>/0010001/wssd0010001_session_1_anat.nii.gz
    <mri-dir>/0010001/swssd0010001_session_1_anat_gm.nii.gz
    <mri-dir>/0010002/...

Uses the SAME core as the API endpoint (MRI_Ingestion.ingest_patient_scans), so a
bulk seed produces identical rows to a live single-patient ingest. Each patient
is committed in its own transaction, so a failure on one patient does not lose
the others.

This is an ADMIN / training tool (not a clinician upload). The MRI data is large
(~1.8 GB for 1-129) and is NOT in the repo - point --mri-dir at wherever it lives
(a local folder, or a path mounted into the container).

Run (from app/, DB up + seeded with patients, .env set):
    python seed_mri.py --mri-dir "<path-to>/NYU_Athena_preproc_1-129"
    python seed_mri.py --mri-dir <path> --skip-existing      # resume a run
    python seed_mri.py --mri-dir <path> --patient 0010001    # one patient
"""

import os
import sys
import argparse
import logging

from database import get_connection
from MRI_Ingestion import (
    ingest_patient_scans, PatientNotFoundError, ScanNotFoundError,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("anima.seed.mri")


def _has_current_scans(cursor, patient_id: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM dbo.MRI WHERE patient_ID = ? AND is_current = 1;",
        patient_id,
    )
    return cursor.fetchone()[0] > 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-ingest training MRI scans.")
    ap.add_argument("--mri-dir", required=True,
                    help="Directory of per-patient MRI folders (named by patient_ID).")
    ap.add_argument("--mode", default="replace", choices=["replace", "new_session"],
                    help="How to handle patients that already have scans.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip patients that already have a current scan (resumable).")
    ap.add_argument("--limit", type=int, default=None, help="Only ingest the first N folders.")
    ap.add_argument("--patient", default=None, help="Ingest a single patient_ID and exit.")
    args = ap.parse_args()

    if not os.path.isdir(args.mri_dir):
        sys.exit(f"ERROR: MRI directory not found: {args.mri_dir}")

    folders = sorted(d for d in os.listdir(args.mri_dir)
                     if os.path.isdir(os.path.join(args.mri_dir, d)))
    if args.patient:
        want = args.patient.strip().split(".")[0].zfill(7)
        folders = [d for d in folders if d.strip().split(".")[0].zfill(7) == want]
    if args.limit:
        folders = folders[: args.limit]
    if not folders:
        sys.exit("No matching patient folders found under --mri-dir.")

    logger.info("Ingesting MRI for %d patient folder(s) from %s (mode=%s) ...",
                len(folders), args.mri_dir, args.mode)

    conn = get_connection()
    ok = skipped = failed = 0
    total_slices = 0
    try:
        cursor = conn.cursor()
        for name in folders:
            patient_id = name.strip().split(".")[0].zfill(7)
            folder = os.path.join(args.mri_dir, name)
            try:
                if args.skip_existing and _has_current_scans(cursor, patient_id):
                    logger.info("  skip %s: already has a current scan", patient_id)
                    skipped += 1
                    continue
                results = ingest_patient_scans(cursor, patient_id, folder, mode=args.mode)
                conn.commit()
                sc = sum(r["slice_count"] for r in results)
                total_slices += sc
                ok += 1
                logger.info("  ok   %s: %d slices across %d scan(s) (session %d)",
                            patient_id, sc, len(results), results[0]["scan_session"])
            except PatientNotFoundError:
                conn.rollback()
                logger.warning("  skip %s: patient not in DB (run seed_dsm5.py first)", patient_id)
                skipped += 1
            except ScanNotFoundError as exc:
                conn.rollback()
                logger.error("  FAIL %s: %s scan not found in folder", patient_id, exc)
                failed += 1
            except Exception:
                conn.rollback()
                logger.exception("  FAIL %s: unexpected error", patient_id)
                failed += 1
    finally:
        conn.close()

    print("\nMRI seed complete.")
    print(f"  Ingested:     {ok}")
    print(f"  Skipped:      {skipped}")
    print(f"  Failed:       {failed}")
    print(f"  Total slices: {total_slices}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
