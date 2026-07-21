"""
Unit tests for MRI ingestion (app/MRI_Ingestion.py).

Builds tiny synthetic NIfTI volumes on disk and runs the REAL ingestion core
(ingest_patient_scans) against the in-memory mock dataset, with the JPEG output
redirected to a temp folder. No SQL Server and no real scans needed.
"""

import os
import numpy as np
import nibabel as nib
import pytest

import MRI_Ingestion
from MRI_Ingestion import (
    ingest_patient_scans, _resolve_session, _locate_nii, _nii_to_slice_stack,
    PatientNotFoundError, ScanNotFoundError,
)

pytestmark = pytest.mark.unit

SHAPE = (8, 8, 5)     # 5 axial slices -> fast


def _write_scans(folder, pid, include_gm=True):
    """Write a synthetic anat (and optionally anat_gm) .nii.gz into ``folder``."""
    os.makedirs(folder, exist_ok=True)
    scans = [("anat", f"wssd{pid}_session_1_anat.nii.gz")]
    if include_gm:
        scans.append(("anat_gm", f"swssd{pid}_session_1_anat_gm.nii.gz"))
    for _scan, fname in scans:
        data = (np.random.rand(*SHAPE) * 100).astype(np.float32)
        nib.save(nib.Nifti1Image(data, affine=np.eye(4)), os.path.join(folder, fname))


@pytest.fixture
def images_dir(tmp_path, monkeypatch):
    """Redirect the processed-JPEG output to a temp folder."""
    out = tmp_path / "mri_images"
    monkeypatch.setattr(MRI_Ingestion, "MRI_IMAGES_DIR", str(out))
    return out


@pytest.fixture
def cursor(fake_conn):
    return fake_conn.cursor()


# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------
def test_first_ingest_is_session_1(cursor):
    assert _resolve_session(cursor, "0010200", "replace") == 1
    assert _resolve_session(cursor, "0010200", "new_session") == 1


def test_new_session_increments(cursor, mockdb):
    mockdb.mri.append({"patient_ID": "0010200", "scan_type": "anat", "file_path": "x",
                       "is_current": 1, "scan_session": 1, "slice_count": 5})
    assert _resolve_session(cursor, "0010200", "new_session") == 2
    assert _resolve_session(cursor, "0010200", "replace") == 1   # reuse current


# ---------------------------------------------------------------------------
# Scan location
# ---------------------------------------------------------------------------
def test_locate_nii_separates_anat_and_gm(tmp_path):
    src = tmp_path / "src"
    _write_scans(str(src), "0010200")
    anat = _locate_nii(str(src), str(tmp_path), gm=False)
    gm = _locate_nii(str(src), str(tmp_path), gm=True)
    assert anat.endswith("_anat.nii.gz") and "anat_gm" not in os.path.basename(anat)
    assert gm.endswith("_anat_gm.nii.gz")


def test_nii_to_slice_stack_writes_named_jpegs(tmp_path):
    src = tmp_path / "src"
    _write_scans(str(src), "0010200")
    nii = _locate_nii(str(src), str(tmp_path), gm=False)
    out = tmp_path / "out"
    os.makedirs(out)
    n = _nii_to_slice_stack(nii, str(out), "0010200", "anat")
    assert n == SHAPE[2]
    files = sorted(os.listdir(out))
    assert files[0] == "0010200_anat_0000.jpg" and len(files) == SHAPE[2]


# ---------------------------------------------------------------------------
# Full ingest core
# ---------------------------------------------------------------------------
def test_ingest_records_two_current_scans(cursor, mockdb, images_dir, tmp_path):
    mockdb.patients["0010200"] = {}                      # patient must exist (FK)
    src = tmp_path / "src"
    _write_scans(str(src), "0010200")

    results = ingest_patient_scans(cursor, "0010200", str(src),
                                   mode="replace", tmp_root=str(tmp_path))
    assert {r["scan_type"] for r in results} == {"anat", "anat_gm"}
    current = mockdb.current_mri("0010200")
    assert len(current) == 2 and all(r["scan_session"] == 1 for r in current)
    # JPEGs actually written for both scans
    for r in results:
        assert r["slice_count"] == SHAPE[2]
        assert len(os.listdir(r["directory"])) == SHAPE[2]


def test_unknown_patient_raises(cursor, images_dir, tmp_path):
    src = tmp_path / "src"
    _write_scans(str(src), "0010999")
    with pytest.raises(PatientNotFoundError):
        ingest_patient_scans(cursor, "0010999", str(src), tmp_root=str(tmp_path))


def test_missing_gm_scan_raises(cursor, mockdb, images_dir, tmp_path):
    mockdb.patients["0010200"] = {}
    src = tmp_path / "src"
    _write_scans(str(src), "0010200", include_gm=False)   # only anat present
    with pytest.raises(ScanNotFoundError) as exc:
        ingest_patient_scans(cursor, "0010200", str(src), tmp_root=str(tmp_path))
    assert str(exc.value) == "anat_gm"


def test_invalid_mode_raises(cursor, mockdb, images_dir, tmp_path):
    mockdb.patients["0010200"] = {}
    src = tmp_path / "src"
    _write_scans(str(src), "0010200")
    with pytest.raises(ValueError):
        ingest_patient_scans(cursor, "0010200", str(src), mode="bogus",
                             tmp_root=str(tmp_path))


def test_new_session_keeps_history(cursor, mockdb, images_dir, tmp_path):
    mockdb.patients["0010200"] = {}
    src = tmp_path / "src"
    _write_scans(str(src), "0010200")
    ingest_patient_scans(cursor, "0010200", str(src), mode="replace", tmp_root=str(tmp_path))
    ingest_patient_scans(cursor, "0010200", str(src), mode="new_session", tmp_root=str(tmp_path))
    current = mockdb.current_mri("0010200")
    history = [r for r in mockdb.mri if r["is_current"] == 0]
    assert len(current) == 2 and all(r["scan_session"] == 2 for r in current)   # new current
    assert len(history) == 2 and all(r["scan_session"] == 1 for r in history)   # old kept


def test_replace_keeps_single_current_session(cursor, mockdb, images_dir, tmp_path):
    mockdb.patients["0010200"] = {}
    src = tmp_path / "src"
    _write_scans(str(src), "0010200")
    ingest_patient_scans(cursor, "0010200", str(src), mode="replace", tmp_root=str(tmp_path))
    ingest_patient_scans(cursor, "0010200", str(src), mode="replace", tmp_root=str(tmp_path))
    current = mockdb.current_mri("0010200")
    assert len(current) == 2 and all(r["scan_session"] == 1 for r in current)   # no history piled up
    assert len([r for r in mockdb.mri if r["is_current"] == 0]) == 0
