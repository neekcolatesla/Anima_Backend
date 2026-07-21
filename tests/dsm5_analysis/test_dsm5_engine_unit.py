"""
Unit tests for the DSM-5 (NLP) analysis engine (app/DSM5_Analysis.py).

Proves the engine works and is connected as intended, WITHOUT a database, the
trained head, or a Bio_ClinicalBERT download:
  * the subtype rules and the heuristic risk mapping;
  * end-to-end analyze_patient wiring against the mock dataset (fetch -> score ->
    persist -> return) on BOTH the heuristic path and the real trained-model path
    (the language features are stubbed so no BERT is needed);
  * the 404 when a patient has no assessment;
  * the FastAPI route is wired to the engine.
"""

import os
import sys
import types

import pytest
import torch

import DSM5_Analysis
from DSM5_Analysis import _predict_subtype, _heuristic_score
import dsm5_model

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------
def test_subtype_rules():
    assert _predict_subtype(70, 70) == "Combined"      # both elevated
    assert _predict_subtype(70, 40) == "Inattentive"   # only inattentive
    assert _predict_subtype(40, 70) == "Hyperactive"   # only hyperactive
    assert _predict_subtype(40, 40) == "None"          # neither


def test_heuristic_risk_tracks_severity():
    assert _heuristic_score(None, None) == 0.0
    low = _heuristic_score(45, 42)
    high = _heuristic_score(85, 80)
    assert 0 <= low < high <= 100


# ---------------------------------------------------------------------------
# Helpers to plug the mock dataset into the engine
# ---------------------------------------------------------------------------
def _seed_patient(db, pid="0010200", age=25, sex=1, is_child=0,
                  inatt=72, hyper=68, notes="reports restlessness and poor focus"):
    db.patients[pid] = {"name": None, "biological_sex": sex, "is_child": is_child, "age": age}
    db.add_assessment(pid, inattentive_score=inatt, hyperactive_score=hyper,
                      clinician_notes=notes)
    return pid


# ---------------------------------------------------------------------------
# Engine wiring — heuristic path (no trained head present)
# ---------------------------------------------------------------------------
def test_analyze_patient_heuristic_path(monkeypatch, mockdb, fake_conn):
    pid = _seed_patient(mockdb)
    monkeypatch.setattr(DSM5_Analysis, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(DSM5_Analysis, "HEAD_PATH", "/no/such/head.pt")

    out = DSM5_Analysis.analyze_patient(pid)

    assert out["status"] == "success"
    assert out["scoring_method"] == "heuristic"
    assert 0 <= out["nlp_risk_score"] <= 100
    assert out["predicted_subtype"] in {"Combined", "Inattentive", "Hyperactive", "None"}
    # explanation present and honest about the path
    ex = out["explanation"]["text_model"]
    assert ex["method"] == "heuristic"
    assert ex["influential_words"] == []          # no model -> no word attribution
    assert ex["feature_importance"]               # still has score-based bars
    # risk persisted back onto the assessment
    assert mockdb.assessment_for(pid)["nlp_risk_score"] == out["nlp_risk_score"]


# ---------------------------------------------------------------------------
# Engine wiring — trained-model path (BERT + features stubbed, real head)
# ---------------------------------------------------------------------------
def test_analyze_patient_model_path(monkeypatch, tmp_path, mockdb, fake_conn):
    pid = _seed_patient(mockdb)

    # A real (tiny, untrained) DSM5Head saved in the shipped {config, state_dict} form.
    head_path = tmp_path / "dsm5_head.pt"
    dsm5_model.save_head(dsm5_model.DSM5Head(input_dim=773, note_pca_k=None), str(head_path))

    # Stub the language features so no Bio_ClinicalBERT is loaded.
    fake_feats = types.ModuleType("dsm5_features")
    fake_feats.STRUCTURED_FEATURE_NAMES = [
        "age", "biological_sex", "is_child", "inattentive_score", "hyperactive_score"]
    fake_feats.build_feature_vector = lambda *a, **k: torch.cat(
        [torch.tensor([25., 1., 0., 72., 68.]), torch.full((768,), 0.1)])
    fake_feats.embed_note_tokens = lambda note: (
        ["restless", "focused"], torch.stack([torch.full((768,), 0.2), torch.full((768,), -0.1)]))
    monkeypatch.setitem(sys.modules, "dsm5_features", fake_feats)

    monkeypatch.setattr(DSM5_Analysis, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(DSM5_Analysis, "HEAD_PATH", str(head_path))

    out = DSM5_Analysis.analyze_patient(pid)

    assert out["status"] == "success"
    assert out["scoring_method"] == "model"       # the head-loader fix is exercised
    ex = out["explanation"]["text_model"]
    assert ex["method"] == "model"
    # feature importance is a ranked, ~100% bar chart; words carry signed pushes
    assert abs(sum(f["impact_percent"] for f in ex["feature_importance"]) - 100.0) < 0.5
    assert ex["influential_words"] and "push" in ex["influential_words"][0]


def test_analyze_patient_404_without_assessment(monkeypatch, mockdb, fake_conn):
    from fastapi import HTTPException
    mockdb.patients["0010999"] = {"biological_sex": 1, "is_child": 0, "age": 30}  # no assessment
    monkeypatch.setattr(DSM5_Analysis, "get_connection", lambda: fake_conn)
    with pytest.raises(HTTPException) as exc:
        DSM5_Analysis.analyze_patient("0010999")
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# The API route is connected to the engine
# ---------------------------------------------------------------------------
def test_endpoint_calls_engine(monkeypatch, mockdb, fake_conn):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    pid = _seed_patient(mockdb)
    monkeypatch.setattr(DSM5_Analysis, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(DSM5_Analysis, "HEAD_PATH", "/no/such/head.pt")

    app = FastAPI()
    app.include_router(DSM5_Analysis.router)
    r = TestClient(app).post(f"/api/analysis/dsm5/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["patient_ID"] == pid and "nlp_risk_score" in body
    assert body["explanation"]["text_model"]["method"] == "heuristic"
