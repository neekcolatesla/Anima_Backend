"""
Anima - MRI ingestion pipeline (longitudinal).

Owns POST /api/ingest/mri. Extracts a patient's MRI ZIP, normalises each
anatomical NIfTI (anat, anat_gm) to 0-255, writes the FULL axial slice stack as
numbered JPEGs into /app/static/mri_images/<patient>/session_<n>/<scan>/, and
records MRI rows under the longitudinal model:

  * mode='replace'      (default) - CORRECTION of the current scan: the previous
                        current row + its slice folder are removed and the new
                        scan becomes current (same scan_session).
  * mode='new_session'  - a genuinely NEW acquisition: the previous current scan
                        is demoted to history (is_current = 0, folder kept) and
                        the new scan becomes current with the next scan_session.

The reusable core (ingest_patient_scans) is framework-agnostic and is called by
BOTH this endpoint and the training seeder (seed_mri.py), so single-patient and
bulk ingestion share one code path.
"""

import os
import glob
import shutil
import zipfile
import logging
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

import pyodbc
import numpy as np
import nibabel as nib
from PIL import Image

from database import get_db

logger = logging.getLogger("anima.ingest.mri")

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

# Processed 2D JPEG slices are written to the dedicated Docker volume mounted at
# /app/static/mri_images (see docker-compose.yml).
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MRI_IMAGES_DIR = os.path.join(STATIC_DIR, "mri_images")

VALID_MODES = ("replace", "new_session")


# =============================================================================
# Framework-agnostic errors (so the seeder can use the core without FastAPI).
# =============================================================================
class PatientNotFoundError(Exception):
    """Raised when the patient does not exist (MRI.patient_ID is an FK)."""


class ScanNotFoundError(Exception):
    """Raised when a required scan (anat / anat_gm) is not in the source."""


# =============================================================================
# Extraction + NIfTI helpers (unchanged, shared by all callers)
# =============================================================================
def _safe_extract(zip_path: str, dest: str) -> None:
    """Extract a ZIP, rejecting any member that would escape ``dest`` (zip-slip)."""
    dest_real = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = os.path.realpath(os.path.join(dest, member))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise HTTPException(status_code=400,
                                    detail=f"Unsafe path in archive: {member}")
        zf.extractall(dest)


def _find_file(root: str, patterns, must_not_contain: Optional[str] = None) -> Optional[str]:
    """Recursively find the first file under ``root`` matching any glob pattern."""
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(root, "**", pat), recursive=True)):
            if os.path.isfile(path):
                base = os.path.basename(path).lower()
                if must_not_contain and must_not_contain in base:
                    continue
                return path
    return None


def _locate_nii(search_root: str, tmp_root: str, gm: bool) -> Optional[str]:
    """Locate a loadable .nii/.nii.gz for a scan type.

    Follows the documented pipeline first (nested *anat.nii.zip / *_anat_gm.nii.zip
    archives, which are extracted to reveal the raw .nii), then falls back to a
    direct .nii.gz / .nii file (the layout of the sample archive and the bulk
    training folders).
    """
    if gm:  # grey-matter scan
        zip_patterns = ["*_anat_gm.nii.zip", "*anat_gm.nii.zip"]
        nii_patterns = ["*_anat_gm.nii.gz", "*anat_gm.nii.gz",
                        "*_anat_gm.nii", "*anat_gm.nii"]
        exclude = None
    else:   # structural anatomical scan - must NOT pick up the _anat_gm files
        zip_patterns = ["*anat.nii.zip"]
        nii_patterns = ["*anat.nii.gz", "*anat.nii"]
        exclude = "anat_gm"

    # 1) documented nested-ZIP path
    nested = _find_file(search_root, zip_patterns, must_not_contain=exclude)
    if nested:
        inner_dir = tempfile.mkdtemp(dir=tmp_root)
        _safe_extract(nested, inner_dir)
        inner = _find_file(inner_dir, ["*.nii.gz", "*.nii"], must_not_contain=exclude)
        if inner:
            return inner

    # 2) fallback: the .nii.gz / .nii is stored directly in the source
    return _find_file(search_root, nii_patterns, must_not_contain=exclude)


def _nii_to_slice_stack(nii_path: str, out_dir: str,
                        patient_id: str, scan_type: str) -> int:
    """Normalise a NIfTI volume to 0-255 and save EVERY axial slice as a numbered
    JPEG into ``out_dir``. Returns the number of slices written.
    """
    img = nib.load(nii_path)
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    while data.ndim > 3:            # drop trailing 4th (time) dimension if present
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}.")

    mn, mx = float(np.nanmin(data)), float(np.nanmax(data))
    span = mx - mn
    n_slices = data.shape[2]                  # axial slices along the last axis
    width = max(4, len(str(n_slices - 1)))    # zero-padded index preserves order

    for i in range(n_slices):
        sl = data[:, :, i]
        norm = ((sl - mn) / span * 255.0) if span > 0 else np.zeros_like(sl)
        norm = np.rot90(np.nan_to_num(norm).astype(np.uint8))  # display orientation
        out_path = os.path.join(out_dir, f"{patient_id}_{scan_type}_{i:0{width}d}.jpg")
        Image.fromarray(norm).convert("L").save(out_path, format="JPEG", quality=90)

    return n_slices


# =============================================================================
# Longitudinal helpers
# =============================================================================
def _patient_exists(cursor, patient_id: str) -> bool:
    cursor.execute("SELECT 1 FROM dbo.Patient WHERE patient_ID = ?;", patient_id)
    return cursor.fetchone() is not None


def _resolve_session(cursor, patient_id: str, mode: str) -> int:
    """Pick the scan_session number for this ingestion.

    First ingest -> 1. new_session -> max existing + 1. replace -> reuse the
    patient's current session (the one being corrected).
    """
    cursor.execute("SELECT MAX(scan_session) FROM dbo.MRI WHERE patient_ID = ?;", patient_id)
    row = cursor.fetchone()
    max_all = row[0] if row else None
    if max_all is None:
        return 1
    if mode == "new_session":
        return int(max_all) + 1
    cursor.execute(
        "SELECT MAX(scan_session) FROM dbo.MRI WHERE patient_ID = ? AND is_current = 1;",
        patient_id,
    )
    cur = cursor.fetchone()[0]
    return int(cur) if cur is not None else int(max_all)


# =============================================================================
# CORE - shared by the API endpoint and the training seeder
# =============================================================================
def ingest_patient_scans(cursor, patient_id: str, source_dir: str,
                         mode: str = "replace", tmp_root: Optional[str] = None) -> list:
    """Locate, slice, and record BOTH anatomical scans for one patient.

    ``source_dir`` is a directory already containing the patient's scan files
    (.nii / .nii.gz, optionally nested .zip). Does NOT commit - the caller owns
    the transaction. Returns a list of dicts (scan_type, directory, slice_count,
    scan_session). Raises PatientNotFoundError / ScanNotFoundError.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if not _patient_exists(cursor, patient_id):
        raise PatientNotFoundError(patient_id)

    tmp_root = tmp_root or tempfile.gettempdir()
    session = _resolve_session(cursor, patient_id, mode)
    results = []

    for scan_type, is_gm in (("anat", False), ("anat_gm", True)):
        nii_path = _locate_nii(source_dir, tmp_root, gm=is_gm)
        if nii_path is None:
            raise ScanNotFoundError(scan_type)

        scan_dir = os.path.join(MRI_IMAGES_DIR, patient_id, f"session_{session}", scan_type)

        # --- longitudinal DB handling for this scan_type ---
        if mode == "replace":
            # Correction: remove the current row(s) + their slice folders, unless
            # the folder is the one we are about to (re)write.
            cursor.execute(
                "SELECT file_path FROM dbo.MRI "
                "WHERE patient_ID = ? AND scan_type = ? AND is_current = 1;",
                patient_id, scan_type,
            )
            for (old_path,) in cursor.fetchall():
                if (old_path and os.path.isdir(old_path)
                        and os.path.realpath(old_path) != os.path.realpath(scan_dir)):
                    shutil.rmtree(old_path, ignore_errors=True)
            cursor.execute(
                "DELETE FROM dbo.MRI "
                "WHERE patient_ID = ? AND scan_type = ? AND is_current = 1;",
                patient_id, scan_type,
            )
        else:  # new_session: demote the old current to history (keep its folder)
            cursor.execute(
                "UPDATE dbo.MRI SET is_current = 0 "
                "WHERE patient_ID = ? AND scan_type = ? AND is_current = 1;",
                patient_id, scan_type,
            )

        # (Re)create a fresh target folder, then write the full slice stack.
        if os.path.isdir(scan_dir):
            shutil.rmtree(scan_dir)
        os.makedirs(scan_dir, exist_ok=True)
        slice_count = _nii_to_slice_stack(nii_path, scan_dir, patient_id, scan_type)

        cursor.execute(
            """
            INSERT INTO dbo.MRI
                (patient_ID, scan_type, file_path, slice_count, is_current, scan_session)
            VALUES (?, ?, ?, ?, 1, ?);
            """,
            patient_id, scan_type, os.path.abspath(scan_dir), slice_count, session,
        )
        results.append({"scan_type": scan_type, "directory": os.path.abspath(scan_dir),
                        "slice_count": slice_count, "scan_session": session})

    return results


# =============================================================================
# API endpoint - single patient
# =============================================================================
@router.post("/mri")
async def ingest_mri(
    file: UploadFile = File(...),
    mode: str = Query("replace", pattern="^(replace|new_session)$",
                      description="replace = correct the current scan; "
                                  "new_session = keep the old scan as history."),
    conn: pyodbc.Connection = Depends(get_db),
) -> dict:
    """Ingest one patient's MRI archive (ZIP named with the 7-digit patient ID).

    The upload is streamed to disk (not buffered in RAM), extracted, and passed to
    the shared core. ``mode`` chooses the longitudinal behaviour (see module docs).
    """
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Upload must be a .zip named with the patient's 7-digit ID "
                   "(e.g. 0010001.zip).",
        )
    patient_id = filename[:-4]

    os.makedirs(MRI_IMAGES_DIR, exist_ok=True)
    cursor = conn.cursor()

    with tempfile.TemporaryDirectory() as tmp:
        # Stream the upload to disk in chunks (avoids holding ~14 MB+ in memory).
        primary_path = os.path.join(tmp, filename)
        with open(primary_path, "wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        if os.path.getsize(primary_path) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        extract_root = os.path.join(tmp, "extracted")
        os.makedirs(extract_root, exist_ok=True)
        try:
            _safe_extract(primary_path, extract_root)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400,
                                detail="Uploaded file is not a valid ZIP archive.")

        try:
            results = ingest_patient_scans(cursor, patient_id, extract_root,
                                           mode=mode, tmp_root=tmp)
            conn.commit()
        except PatientNotFoundError:
            conn.rollback()
            raise HTTPException(
                status_code=404,
                detail=f"Patient '{patient_id}' not found. Ingest demographics first.",
            )
        except ScanNotFoundError as exc:
            conn.rollback()
            raise HTTPException(
                status_code=422,
                detail=f"Could not find the {exc} scan in the archive.",
            )
        except HTTPException:
            conn.rollback()
            raise
        except Exception as exc:
            conn.rollback()
            logger.exception("MRI ingestion failed.")
            raise HTTPException(status_code=422, detail=f"MRI ingestion failed: {exc}")

    total_slices = sum(r["slice_count"] for r in results)
    session = results[0]["scan_session"] if results else None
    return {
        "status": "success",
        "patient_ID": patient_id,
        "mode": mode,
        "scan_session": session,
        "scans": results,
        "total_slices": total_slices,
        "message": (f"Stored {total_slices} MRI slice image(s) across "
                    f"{len(results)} scan(s) for patient {patient_id} "
                    f"(session {session}, mode={mode})."),
    }
