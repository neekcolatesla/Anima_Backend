"""
Unit tests for the explainability (xAI) layer, WITHOUT a database, the trained
models, or a Bio_ClinicalBERT download.

The xAI layer is cross-cutting: every engine attaches an ``explanation`` block so
the Power Apps frontend can show WHY a score came out the way it did. These tests
prove each explanation is well-formed and honest in isolation:

  * DSM-5 text model  - per-feature importance bars + per-word "push" attribution
    (linear-head contributions) and the plain-language summary;
  * MRI image model   - the Grad-CAM pipeline: slice-index parsing, blue->red
    colour map, PNG data-URI encoding, and the overlay produced from a tiny CNN;
  * Combined engine   - that BOTH channels' explanations are merged, and that a
    missing/untrained image channel degrades to an honest ``available: False``
    block rather than a broken/missing one.
"""

import sys
import types
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi import HTTPException

pytestmark = pytest.mark.unit


# ===========================================================================
# DSM-5 text model: feature importance + influential words + summary
# ===========================================================================
import DSM5_Analysis
import dsm5_model


def _valid_feature_importance(items):
    """A ranked, ~100%% set of signed bars."""
    assert items, "feature_importance must not be empty"
    for f in items:
        assert set(f) >= {"feature", "impact_percent", "direction"}
        assert 0.0 <= f["impact_percent"] <= 100.0
        assert f["direction"] in {"toward ADHD", "toward control"}
    # ranked by impact, descending
    impacts = [f["impact_percent"] for f in items]
    assert impacts == sorted(impacts, reverse=True)
    # bars are percentages of the total -> sum to ~100
    assert abs(sum(impacts) - 100.0) < 0.5


def test_heuristic_explanation_is_honest_and_wordless():
    ex = DSM5_Analysis._heuristic_explanation(inattentive_score=72, hyperactive_score=40)
    assert ex["method"] == "heuristic"
    assert ex["influential_words"] == []          # no trained model -> no word attribution
    _valid_feature_importance(ex["feature_importance"])
    assert "trained text model not loaded" in ex["summary"]


def test_text_summary_names_top_driver_and_words():
    assert DSM5_Analysis._text_summary([], []) == "No explanation available."
    fi = [{"feature": "Inattentive score", "impact_percent": 61.0, "direction": "toward ADHD"},
          {"feature": "Clinical notes", "impact_percent": 39.0, "direction": "toward control"}]
    words = [{"word": "restless", "push": 88.0, "direction": "toward ADHD"},
             {"word": "calm", "push": -40.0, "direction": "toward control"}]
    msg = DSM5_Analysis._text_summary(fi, words)
    assert "Inattentive score" in msg and "61.0%" in msg
    assert "restless" in msg and "calm" not in msg    # only 'toward ADHD' words are surfaced


def test_explain_text_ranks_features_and_scores_words():
    """_explain_text on a real (tiny) linear head: exact per-feature + per-word maths."""
    # stub the language features so no BERT is needed
    fake_feats = types.ModuleType("dsm5_features")
    fake_feats.STRUCTURED_FEATURE_NAMES = [
        "age", "biological_sex", "is_child", "inattentive_score", "hyperactive_score"]
    fake_feats.embed_note_tokens = lambda note: (
        ["restless", "distractible", "the"],
        torch.stack([torch.full((768,), 0.3), torch.full((768,), -0.2), torch.zeros(768)]))
    monkey = pytest.MonkeyPatch()
    monkey.setitem(sys.modules, "dsm5_features", fake_feats)
    try:
        head = dsm5_model.DSM5Head(input_dim=773, note_pca_k=None)   # structured_dim=5
        features = torch.cat([torch.tensor([25., 1., 0., 72., 68.]), torch.full((768,), 0.1)])
        ex = DSM5_Analysis._explain_text(head, features, "patient reports restless behaviour")
    finally:
        monkey.undo()

    assert ex["method"] == "model"
    _valid_feature_importance(ex["feature_importance"])
    # the note is folded into a single "Clinical notes" bar alongside the 5 structured ones
    names = {f["feature"] for f in ex["feature_importance"]}
    assert "Clinical notes" in names and len(ex["feature_importance"]) == 6
    # words carry a signed, bounded push and are ranked by absolute strength
    words = ex["influential_words"]
    assert words, "model path must attribute words"
    for w in words:
        assert set(w) >= {"word", "push", "direction"}
        assert -100.0 <= w["push"] <= 100.0
        assert w["direction"] in {"toward ADHD", "toward control"}
    pushes = [abs(w["push"]) for w in words]
    assert pushes == sorted(pushes, reverse=True)


# ===========================================================================
# MRI image model: the Grad-CAM heatmap pipeline
# ===========================================================================
import MRI_Analysis
import mri_model


def test_slice_number_parses_axial_index():
    assert MRI_Analysis._slice_number("0010001_anat_0094.jpg") == 94
    assert MRI_Analysis._slice_number("/x/y/0010001_anat_gm_0007.jpg") == 7
    assert MRI_Analysis._slice_number("nonsense.jpg") == -1     # no numeric tail


def test_heat_colors_maps_cold_blue_to_hot_red():
    heat = MRI_Analysis._heat_colors(np.array([0.0, 1.0]))
    assert heat.dtype == np.uint8 and heat.shape == (2, 3)
    cold, hot = heat[0], heat[1]
    assert cold[2] > cold[0]      # cold end is blue-dominant (B > R)
    assert hot[0] > hot[2]        # hot end is red-dominant  (R > B)


def test_to_data_uri_is_a_decodable_png():
    rgb = (np.random.rand(24, 24, 3) * 255).astype(np.uint8)
    uri = MRI_Analysis._to_data_uri(rgb)
    assert uri.startswith("data:image/png;base64,")
    img = Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert img.format == "PNG" and img.size == (24, 24)


def test_gradcam_overlay_shape_and_type():
    """A real (tiny, untrained) CNN + a 2-channel slice -> an (H,W,3) uint8 overlay."""
    size = 32
    model = mri_model.MRICNN(input_size=size)
    slice_tensor = torch.rand(2, size, size)          # (anat, anat_gm)
    overlay = MRI_Analysis._gradcam_overlay(model, slice_tensor)
    assert overlay.shape == (size, size, 3)
    assert overlay.dtype == np.uint8
    assert overlay.max() <= 255 and overlay.min() >= 0


# --- full engine: the explanation block analyze_mri returns --------------------
def _write_slices(folder, pid, scan, n=6, size=32):
    import os
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        arr = (np.random.rand(size, size) * 255).astype(np.uint8)
        Image.fromarray(arr).convert("L").save(
            os.path.join(folder, f"{pid}_{scan}_{i:04d}.jpg"), "JPEG")
    return folder


@pytest.fixture
def mri_patient(tmp_path, mockdb, monkeypatch):
    pid = "0010200"
    anat = _write_slices(tmp_path / "anat", pid, "anat")
    gm = _write_slices(tmp_path / "anat_gm", pid, "anat_gm")
    mockdb.patients[pid] = {"biological_sex": 1, "is_child": 0, "age": 25}
    mockdb.add_mri(pid, "anat", str(anat), is_current=1)
    mockdb.add_mri(pid, "anat_gm", str(gm), is_current=1)
    cnn_path = tmp_path / "mri_cnn.pt"
    mri_model.save_cnn(mri_model.MRICNN(input_size=32), str(cnn_path))
    monkeypatch.setattr(MRI_Analysis, "CNN_PATH", str(cnn_path))
    monkeypatch.setattr(MRI_Analysis, "MRI_CENTRAL_FRAC", 1.0)
    monkeypatch.setattr(MRI_Analysis, "MRI_MIN_FOREGROUND", 0.0)
    monkeypatch.setattr(MRI_Analysis, "MRI_MAX_SLICES", 0)
    return pid


def test_analyze_mri_explanation_block(monkeypatch, mockdb, fake_conn, mri_patient):
    monkeypatch.setattr(MRI_Analysis, "get_connection", lambda: fake_conn)
    ex = MRI_Analysis.analyze_mri(mri_patient)["explanation"]["mri_model"]
    assert ex["available"] is True
    assert isinstance(ex["top_slice_index"], int)
    # heatmap is a Power-Apps-ready PNG data URI that actually decodes
    assert ex["heatmap_image"].startswith("data:image/png;base64,")
    Image.open(BytesIO(base64.b64decode(ex["heatmap_image"].split(",", 1)[1]))).verify()
    assert str(ex["top_slice_index"]) in ex["summary"]     # summary names the driving slice


# ===========================================================================
# Combined engine: merge both explanations; degrade the image channel honestly
# ===========================================================================
import Combined_Analysis as C


class _AuditConn:
    """Swallows the Analysis_Result audit write so analyze_combined can run."""
    class _Cur:
        def execute(self, sql, *a):
            self._r = (1,) if "TOP 1" in " ".join(sql.split()) else None
        def fetchone(self):
            return getattr(self, "_r", None)
    def cursor(self):
        return self._Cur()
    def commit(self):
        pass
    def rollback(self):
        pass
    def close(self):
        pass


_NLP = {"nlp_risk_score": 80.0, "predicted_subtype": "Inattentive",
        "explanation": {"text_model": {
            "method": "model", "summary": "biggest factor ...",
            "feature_importance": [{"feature": "Inattentive score",
                                    "impact_percent": 100.0, "direction": "toward ADHD"}],
            "influential_words": [{"word": "restless", "push": 90.0,
                                   "direction": "toward ADHD"}]}}}
_MRI_OK = {"status": "success", "mri_risk_score": 40.0,
           "explanation": {"mri_model": {"available": True, "top_slice_index": 90,
                                         "heatmap_image": "data:image/png;base64,AAAA",
                                         "summary": "driven by slice 90 ..."}}}


def _stub(monkeypatch, nlp, mri=None, mri_exc=None):
    dsm = types.ModuleType("DSM5_Analysis"); dsm.analyze_patient = lambda pid: nlp
    m = types.ModuleType("MRI_Analysis")
    if mri_exc is not None:
        def _raise(pid): raise mri_exc
        m.analyze_mri = _raise
    else:
        m.analyze_mri = lambda pid: mri
    monkeypatch.setitem(sys.modules, "DSM5_Analysis", dsm)
    monkeypatch.setitem(sys.modules, "MRI_Analysis", m)
    monkeypatch.setattr(C, "get_connection", lambda: _AuditConn())


def test_combined_merges_both_explanations(monkeypatch):
    _stub(monkeypatch, _NLP, _MRI_OK)
    ex = C.analyze_combined("0010200", created_by="A000001")["explanation"]
    # text channel carried through intact
    assert ex["text_model"]["method"] == "model"
    assert ex["text_model"]["influential_words"][0]["word"] == "restless"
    # image channel carried through intact
    assert ex["mri_model"]["available"] is True
    assert ex["mri_model"]["heatmap_image"].startswith("data:image/png")
    assert ex["mri_model"]["top_slice_index"] == 90


def test_combined_degrades_image_channel_when_pending(monkeypatch):
    _stub(monkeypatch, _NLP, {"status": "pending", "mri_risk_score": None})
    ex = C.analyze_combined("0010200")["explanation"]
    assert ex["text_model"]["method"] == "model"          # text still fully explained
    assert ex["mri_model"] == {"available": False, "reason": "pending"}


def test_combined_degrades_image_channel_when_no_scans(monkeypatch):
    _stub(monkeypatch, _NLP, mri_exc=HTTPException(status_code=404, detail="no scans"))
    ex = C.analyze_combined("0010200")["explanation"]
    assert ex["mri_model"] == {"available": False, "reason": "no_scans"}
