"""
Anima - MRI ingestion pipeline.

Owns the /api/ingest/mri endpoint: extracts a patient's MRI ZIP, normalises each
anatomical NIfTI (anat, anat_gm) to 0-255, writes the FULL axial slice stack as
numbered JPEGs into /app/static/mri_images/<patient>/<scan>/, and records one MRI
row per scan folder. (SysArchitecture: "api/ingest - Data Pipelines".)
"""

import os
import glob
import shutil
import zipfile
import logging
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

import pyodbc
import numpy as np
import nibabel as nib
from PIL import Image

from database import get_db

logger = logging.getLogger("anima.ingest.mri")

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

# Processed 2D JPEG slices are written to the dedicated Docker volume mounted at
# /app/static/mri_images (see docker-compose.yml). Resolved relative to this
# module so it matches main.py's static mount without a cross-import.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MRI_IMAGES_DIR = os.path.join(STATIC_DIR, "mri_images")


def _safe_extract(zip_path: str, dest: str) -> None:
    """Extract a ZIP, rejecting any member that would escape ``dest`` (zip-slip)."""
    dest_real = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = os.path.realpath(os.path.join(dest, member))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsafe path in archive: {member}",
                )
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
    direct .nii.gz / .nii file (the layout of the sample archive).
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

    # 2) fallback: the .nii.gz / .nii is stored directly in the primary archive
    return _find_file(search_root, nii_patterns, must_not_contain=exclude)


def _nii_to_slice_stack(nii_path: str, out_dir: str,
                        patient_id: str, scan_type: str) -> int:
    """Normalise a NIfTI volume to 0-255 and save EVERY axial slice as a
    numbered JPEG into ``out_dir``. Returns the number of slices written.

    The full stack (not a single slice) preserves whole-brain coverage of ADHD
    grey-matter biomarkers - prefrontal cortex, basal ganglia, cerebellum - which
    are distributed through the volume, giving the ML model many data points per
    patient and enabling the clinician-facing 3D/heat-map visualisation.
    """
    img = nib.load(nii_path)
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    while data.ndim > 3:            # drop trailing 4th (time) dimension if present
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}.")

    # Volume-wide 0-255 normalisation so intensities are consistent across slices.
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


@router.post("/mri")
async def ingest_mri(file: UploadFile = File(...),
                     conn: pyodbc.Connection = Depends(get_db)) -> dict:
    """Ingest a patient's MRI archive: extract, normalise, slice the FULL stack.

    The upload must be a ZIP named with the patient's 7-digit ID (e.g.
    0010001.zip). Both anatomical scans (anat, anat_gm) are located and normalised
    to 0-255, then EVERY axial slice is written as a numbered JPEG into a
    patient/scan-specific folder: /app/static/mri_images/<patient>/<scan>/. One
    MRI row is recorded per scan *folder* (file_path = the directory) so the
    analysis engine can read the whole slice stack per scan.
    """
    # Step 1-2: derive patient_ID from the ZIP filename (strip '.zip').
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Upload must be a .zip named with the patient's 7-digit ID "
                   "(e.g. 0010001.zip).",
        )
    patient_id = filename[:-4]

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Verify the patient exists BEFORE processing (MRI.patient_ID is a FK to
    # Patient) - avoids writing orphan JPEGs for an unknown patient.
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM dbo.Patient WHERE patient_ID = ?;", patient_id)
    if cursor.fetchone() is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found. Ingest demographics first.",
        )

    os.makedirs(MRI_IMAGES_DIR, exist_ok=True)
    results = []  # (scan_type, absolute_dir_path, slice_count)

    # Step 3-5 + image processing inside a self-cleaning temp directory.
    with tempfile.TemporaryDirectory() as tmp:
        primary_path = os.path.join(tmp, filename)
        with open(primary_path, "wb") as fh:
            fh.write(raw)

        extract_root = os.path.join(tmp, "extracted")
        os.makedirs(extract_root, exist_ok=True)
        try:
            _safe_extract(primary_path, extract_root)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400,
                                detail="Uploaded file is not a valid ZIP archive.")

        for scan_type, is_gm in (("anat", False), ("anat_gm", True)):
            nii_path = _locate_nii(extract_root, tmp, gm=is_gm)
            if nii_path is None:
                pat = "*_anat_gm.nii" if is_gm else "*anat.nii"
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not find the {scan_type} scan ({pat}[.zip/.gz]) "
                           "in the archive.",
                )

            # Patient- and scan-specific folder; wiped first so a re-ingest never
            # mixes fresh slices with stale ones from a previous run.
            scan_dir = os.path.join(MRI_IMAGES_DIR, patient_id, scan_type)
            if os.path.isdir(scan_dir):
                shutil.rmtree(scan_dir)
            os.makedirs(scan_dir, exist_ok=True)

            try:
                slice_count = _nii_to_slice_stack(nii_path, scan_dir,
                                                  patient_id, scan_type)
            except Exception as exc:
                logger.exception("NIfTI->JPEG stack conversion failed.")
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to process {scan_type} scan: {exc}",
                )
            results.append((scan_type, os.path.abspath(scan_dir), slice_count))

    total_slices = sum(count for _s, _d, count in results)

    # DB: store ONE row per scan *folder* (file_path = the directory), not per
    # image. Delete-then-insert makes re-ingesting a patient's MRI idempotent.
    try:
        cursor.execute("DELETE FROM dbo.MRI WHERE patient_ID = ?;", patient_id)
        for scan_type, dir_path, slice_count in results:
            cursor.execute(
                """
                INSERT INTO dbo.MRI (patient_ID, scan_type, file_path, slice_count)
                VALUES (?, ?, ?, ?);
                """,
                patient_id, scan_type, dir_path, slice_count,
            )
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except pyodbc.Error as exc:
        conn.rollback()
        logger.exception("MRI ingestion DB error.")
        raise HTTPException(status_code=400, detail=f"MRI ingestion failed: {exc}")

    return {
        "status": "success",
        "message": (f"Stored {total_slices} MRI slice image(s) across "
                    f"{len(results)} scan folder(s) for patient {patient_id}."),
        "patient_ID": patient_id,
        "scans": [
            {"scan_type": s, "directory": d, "slice_count": c}
            for s, d, c in results
        ],
        "stored_count": len(results),
        "total_slices": total_slices,
    }