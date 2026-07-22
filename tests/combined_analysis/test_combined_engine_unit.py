"""
Unit tests for the Combined analysis engine (app/Combined_Analysis.py).

Proves the fusion layer works and is connected as intended, WITHOUT a database or
the real component models: the DSM-5 and MRI engines are stubbed so we test the
combined logic in isolation - fusion maths, graceful fallbacks, the append-only
audit row + created_by, RBAC on the trigger, and the RBAC-gated retrieval reads.
"""

import sys
import types
import datetime
import decimal

import pytest
from fastapi import HTTPException

import Combined_Analysis as C

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fusion maths
# ---------------------------------------------------------------------------
def test_fuse_weighted_average():
    score, note = C._fuse(80.0, 40.0)          # 0.7*80 + 0.3*40 = 68
    assert score == 68.0 and "nlp=0.7" in note


def test_fuse_falls_back_to_single_channel():
    assert C._fuse(80.0, None) == (80.0, "nlp=1.0 (mri unavailable)")
    assert C._fuse(None, 40.0)[0] == 40.0
    assert C._fuse(None, None)[0] is None


# ---------------------------------------------------------------------------
# analyze_combined: stub the two component engines + a capturing audit conn
# ---------------------------------------------------------------------------
class _AuditCur:
    def __init__(self, sink):
        self.sink = sink
        self._r = None

    def execute(self, sql, *a):
        q = " ".join(sql.split())
        if "SELECT TOP 1 assessment_ID" in q:
            self._r = (555,)
        elif "SELECT TOP 1 MRI_ID" in q:
            self._r = (777,)
        elif q.startswith("INSERT INTO dbo.Analysis_Result"):
            self.sink.append(a)
            self._r = None
        else:
            self._r = None

    def fetchone(self):
        return self._r


class _AuditConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return _AuditCur(self.sink)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture
def audit_sink(monkeypatch):
    """Capture Analysis_Result inserts; returns the list of insert param-tuples."""
    sink = []
    monkeypatch.setattr(C, "get_connection", lambda: _AuditConn(sink))
    return sink


def _stub_components(monkeypatch, nlp_result, mri_result=None, mri_exc=None):
    dsm = types.ModuleType("DSM5_Analysis")
    dsm.analyze_patient = lambda pid: nlp_result
    mri = types.ModuleType("MRI_Analysis")
    if mri_exc is not None:
        def _raise(pid):
            raise mri_exc
        mri.analyze_mri = _raise
    else:
        mri.analyze_mri = lambda pid: mri_result
    monkeypatch.setitem(sys.modules, "DSM5_Analysis", dsm)
    monkeypatch.setitem(sys.modules, "MRI_Analysis", mri)


_NLP = {"nlp_risk_score": 80.0, "predicted_subtype": "Inattentive",
        "explanation": {"text_model": {"method": "model", "summary": "...",
                                       "feature_importance": [], "influential_words": []}}}
_MRI_OK = {"status": "success", "mri_risk_score": 40.0,
           "explanation": {"mri_model": {"available": True, "top_slice_index": 90,
                                         "heatmap_image": "data:image/png;base64,AAA"}}}


def test_analyze_combined_fuses_both_and_writes_audit(monkeypatch, audit_sink):
    _stub_components(monkeypatch, _NLP, _MRI_OK)
    out = C.analyze_combined("0010200", created_by="A000001")

    assert out["final_combined_score"] == 68.0
    assert out["predicted_diagnosis"] == "ADHD"
    assert out["mri_status"] == "success"
    # explanation merges both channels
    assert out["explanation"]["text_model"]["method"] == "model"
    assert out["explanation"]["mri_model"]["heatmap_image"].startswith("data:image/png")
    # one audit row, carrying created_by + both component IDs/scores
    assert len(audit_sink) == 1
    row = audit_sink[0]
    assert row[0] == "0010200" and row[1] == 555 and row[2] == 777
    assert row[3] == 80.0 and row[4] == 40.0 and row[5] == 68.0 and row[8] == "A000001"


def test_analyze_combined_mri_pending_falls_back_to_nlp(monkeypatch, audit_sink):
    _stub_components(monkeypatch, _NLP, {"status": "pending", "mri_risk_score": None})
    out = C.analyze_combined("0010200")
    assert out["final_combined_score"] == 80.0        # NLP only
    assert out["mri_status"] == "pending"
    assert out["explanation"]["mri_model"] == {"available": False, "reason": "pending"}
    assert audit_sink[0][4] is None                   # mri_risk_score NULL in audit


def test_analyze_combined_no_scans(monkeypatch, audit_sink):
    _stub_components(monkeypatch, _NLP, mri_exc=HTTPException(status_code=404, detail="no scans"))
    out = C.analyze_combined("0010200")
    assert out["mri_status"] == "no_scans"
    assert out["explanation"]["mri_model"]["reason"] == "no_scans"


def test_analyze_combined_propagates_dsm5_404(monkeypatch, audit_sink):
    def _raise(pid):
        raise HTTPException(status_code=404, detail="no assessment")
    dsm = types.ModuleType("DSM5_Analysis"); dsm.analyze_patient = _raise
    mri = types.ModuleType("MRI_Analysis"); mri.analyze_mri = lambda pid: _MRI_OK
    monkeypatch.setitem(sys.modules, "DSM5_Analysis", dsm)
    monkeypatch.setitem(sys.modules, "MRI_Analysis", mri)
    with pytest.raises(HTTPException) as exc:
        C.analyze_combined("9999999")
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# RBAC: _authorize_patient_access for every role
# ---------------------------------------------------------------------------
class _RbacCur:
    def __init__(self, role=None, patient=None, assigned=False):
        self.role, self.patient, self.assigned = role, patient, assigned
        self._r = None

    def execute(self, sql, *a):
        q = " ".join(sql.split())
        if "SELECT role FROM dbo.Users" in q:
            self._r = (self.role,) if self.role else None
        elif "FROM dbo.Patient WHERE patient_ID" in q:
            self._r = self.patient                      # (user_ID, guardian_ID) or None
        elif "FROM dbo.Clinician_Patient_Assignment" in q:
            self._r = (1,) if self.assigned else None
        else:
            self._r = None

    def fetchone(self):
        return self._r


def _authz(**kw):
    requester = kw.pop("requester", "X")
    C._authorize_patient_access(_RbacCur(**kw), requester, "0010200")


@pytest.mark.parametrize("kw", [
    dict(role="Admin", requester="A000001"),                                  # admin: all
    dict(role="Clinician", requester="C0010005", patient=("P0010200", None), assigned=True),
    dict(role="Guardian", requester="G0010003", patient=(None, "0010003")),
    dict(role="Patient", requester="P0010200", patient=("P0010200", None)),
])
def test_authorized_roles_allowed(kw):
    _authz(**kw)          # must not raise


@pytest.mark.parametrize("kw,code", [
    (dict(role="Clinician", requester="C0010009", patient=("P0010200", None), assigned=False), 403),
    (dict(role="Guardian", requester="G0010003", patient=(None, "0010099")), 403),
    (dict(role="Patient", requester="P0010200", patient=("P0010002", None)), 403),
    (dict(role=None, requester="Z9999999"), 401),
])
def test_unauthorized_denied(kw, code):
    with pytest.raises(HTTPException) as exc:
        _authz(**kw)
    assert exc.value.status_code == code


# ---------------------------------------------------------------------------
# POST endpoint stamps created_by after authorising
# ---------------------------------------------------------------------------
def test_endpoint_authorises_then_stamps_created_by(monkeypatch):
    captured = {}
    monkeypatch.setattr(C, "analyze_combined",
                        lambda pid, created_by=None: captured.update(pid=pid, by=created_by) or {"ok": 1})

    class _Conn:
        def cursor(self):
            return _RbacCur(role="Admin")
    out = C.analyze_combined_endpoint("0010200", "A000001", conn=_Conn())
    assert captured == {"pid": "0010200", "by": "A000001"} and out == {"ok": 1}


def test_endpoint_blocks_unassigned_clinician(monkeypatch):
    monkeypatch.setattr(C, "analyze_combined", lambda *a, **k: pytest.fail("should not run"))

    class _Conn:
        def cursor(self):
            return _RbacCur(role="Clinician", patient=("P0010200", None), assigned=False)
    with pytest.raises(HTTPException) as exc:
        C.analyze_combined_endpoint("0010200", "C0010009", conn=_Conn())
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# Retrieval reads (latest + history), RBAC-gated + serialised
# ---------------------------------------------------------------------------
_ROW = (5, "0010200", 555, 777, decimal.Decimal("93.34"), decimal.Decimal("40.0"),
        decimal.Decimal("77.3"), "Combined", "combined_v1",
        datetime.datetime(2026, 7, 21, 14, 0), "A000001")


class _ReadCur(_RbacCur):
    def __init__(self, rows, **kw):
        super().__init__(**kw)
        self.rows = rows
        self._many = []

    def execute(self, sql, *a):
        q = " ".join(sql.split())
        if "FROM dbo.Analysis_Result" in q and "TOP 1" in q:
            self._r = self.rows[0] if self.rows else None
        elif "FROM dbo.Analysis_Result" in q:
            self._many = self.rows
        else:
            super().execute(sql, *a)

    def fetchall(self):
        return self._many


class _ReadConn:
    def __init__(self, cur):
        self._c = cur

    def cursor(self):
        return self._c


def test_get_latest_result_serialises_and_gates(monkeypatch):
    r = C.get_latest_result("0010200", "A000001",
                            conn=_ReadConn(_ReadCur([_ROW], role="Admin")))
    res = r["result"]
    assert res["final_combined_score"] == 77.3 and res["created_by"] == "A000001"
    assert res["created_at"].startswith("2026-07-21")     # datetime -> ISO string


def test_get_history_returns_all(monkeypatch):
    r = C.get_result_history("0010200", "A000001",
                             conn=_ReadConn(_ReadCur([_ROW, _ROW, _ROW], role="Admin")))
    assert r["count"] == 3


def test_retrieval_denies_unauthorised(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        C.get_latest_result("0010200", "C9999999",
                            conn=_ReadConn(_ReadCur([_ROW], role=None)))
    assert exc.value.status_code == 401


def test_retrieval_404_when_no_results(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        C.get_latest_result("0010200", "A000001",
                            conn=_ReadConn(_ReadCur([], role="Admin")))
    assert exc.value.status_code == 404
