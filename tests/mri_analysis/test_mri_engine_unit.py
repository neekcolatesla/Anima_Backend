"""
Unit tests for the MRI analysis engine (app/MRI_Analysis.py).

Proves the engine works and is connected as intended, WITHOUT a database or the
real cohort: a tiny (untrained) MRICNN is saved to disk, synthetic slice JPEGs
stand in for a patient's anat / anat_gm folders, and the engine runs end-to-end
against the mock dataset (fetch current scans -> score -> heatmap -> persist).
"""

import os
import numpy as np
from PIL import Image
import pytest

import MRI_Analysis
import mri_model

pytestmark = pytest.mark.unit


def _write_slices(folder, pid, scan, n=6, size=32):
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        arr = (np.random.rand(size, size) * 255).astype(np.uint8)
        Image.fromarray(arr).convert("L").save(
            os.path.join(folder, f"{pid}_{scan}_{i:04d}.jpg"), "JPEG")
    return folder


@pytest.fixture
def trained_patient(tmp_path, mockdb, monkeypatch):
    """A mock patient with two current scan folders + a saved (tiny) CNN."""
    pid = "0010200"
    anat = _write_slices(tmp_path / "anat", pid, "anat")
    gm = _write_slices(tmp_path / "anat_gm", pid, "anat_gm")
    mockdb.patients[pid] = {"biological_sex": 1, "is_child": 0, "age": 25}
    mockdb.add_mri(pid, "anat", str(anat), is_current=1)
    mockdb.add_mri(pid, "anat_gm", str(gm), is_current=1)

    cnn_path = tmp_path / "mri_cnn.pt"
    mri_model.save_cnn(mri_model.MRICNN(input_size=32), str(cnn_path))
    monkeypatch.setattr(MRI_Analysis, "CNN_PATH", str(cnn_path))
    # use every slice, no foreground filtering (deterministic for the tiny fixture)
    monkeypatch.setattr(MRI_Analysis, "MRI_CENTRAL_FRAC", 1.0)
    monkeypatch.setattr(MRI_Analysis, "MRI_MIN_FOREGROUND", 0.0)
    monkeypatch.setattr(MRI_Analysis, "MRI_MAX_SLICES", 0)
    return pid


def test_analyze_mri_success_with_heatmap(monkeypatch, mockdb, fake_conn, trained_patient):
    monkeypatch.setattr(MRI_Analysis, "get_connection", lambda: fake_conn)
    out = MRI_Analysis.analyze_mri(trained_patient)

    assert out["status"] == "success"
    assert out["predicted_diagnosis"] in {"ADHD", "Control"}
    assert 0 <= out["mri_risk_score"] <= 100
    assert 0 <= out["confidence"] <= 100
    # XAI heatmap block
    mri = out["explanation"]["mri_model"]
    assert mri["available"] is True
    assert isinstance(mri["top_slice_index"], int)
    assert mri["heatmap_image"].startswith("data:image/png;base64,")
    # risk persisted onto BOTH current MRI rows
    for r in mockdb.mri:
        assert r["mri_risk_score"] == out["mri_risk_score"]


def test_diagnosis_follows_threshold(monkeypatch, mockdb, fake_conn, trained_patient):
    monkeypatch.setattr(MRI_Analysis, "get_connection", lambda: fake_conn)
    out = MRI_Analysis.analyze_mri(trained_patient)
    expected = "ADHD" if out["mri_risk_score"] >= MRI_Analysis.MRI_DIAGNOSIS_THRESHOLD else "Control"
    assert out["predicted_diagnosis"] == expected


def test_pending_when_no_trained_weights(monkeypatch, mockdb, fake_conn, trained_patient):
    monkeypatch.setattr(MRI_Analysis, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(MRI_Analysis, "CNN_PATH", "/no/such/mri_cnn.pt")
    out = MRI_Analysis.analyze_mri(trained_patient)
    assert out["status"] == "pending"
    assert out["mri_risk_score"] is None
    # nothing persisted on the pending path
    assert all(r["mri_risk_score"] is None for r in mockdb.mri)


def test_404_when_patient_missing(monkeypatch, mockdb, fake_conn):
    from fastapi import HTTPException
    monkeypatch.setattr(MRI_Analysis, "get_connection", lambda: fake_conn)
    with pytest.raises(HTTPException) as exc:
        MRI_Analysis.analyze_mri("9999999")
    assert exc.value.status_code == 404


def test_404_when_a_scan_type_missing(monkeypatch, tmp_path, mockdb, fake_conn):
    from fastapi import HTTPException
    pid = "0010201"
    mockdb.patients[pid] = {"biological_sex": 0, "is_child": 1, "age": 12}
    mockdb.add_mri(pid, "anat", str(_write_slices(tmp_path / "a", pid, "anat")), is_current=1)
    # no anat_gm -> engine must refuse
    monkeypatch.setattr(MRI_Analysis, "get_connection", lambda: fake_conn)
    with pytest.raises(HTTPException) as exc:
        MRI_Analysis.analyze_mri(pid)
    assert exc.value.status_code == 404


def test_endpoint_calls_engine(monkeypatch, mockdb, fake_conn, trained_patient):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setattr(MRI_Analysis, "get_connection", lambda: fake_conn)
    app = FastAPI()
    app.include_router(MRI_Analysis.router)
    r = TestClient(app).post(f"/api/analysis/mri/{trained_patient}")
    assert r.status_code == 200
    assert r.json()["patient_ID"] == trained_patient
