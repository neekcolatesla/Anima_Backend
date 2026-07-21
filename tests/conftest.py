"""
Shared pytest fixtures for the Anima test suite.

Two jobs:
  1. make the app package importable (it uses flat imports like `import database`);
  2. provide an in-memory **mock database** so the unit tests can add mock data
     and assert on it WITHOUT a real SQL Server.

The mock DB (`MockDB` + `FakeCursor`) emulates just the SQL the ingestion code
issues (patient existence, Patient/DSM5_Assessment/MRI inserts, session lookups),
so tests exercise the real ingestion functions against a fake dataset.
"""

import os
import sys
import types

# --- 1. make app/ importable -------------------------------------------------
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# --- 2. pyodbc fallback ------------------------------------------------------
# The app modules import pyodbc at module load. The UNIT tests never open a real
# connection (they mock it), so if the SQL driver isn't installed we stub pyodbc
# just enough to let imports + `except pyodbc.Error` succeed. When real pyodbc is
# present (your venv) this block does nothing and the live tests use it.
try:
    import pyodbc  # noqa: F401
except Exception:  # pragma: no cover
    _stub = types.ModuleType("pyodbc")

    class _Error(Exception):
        pass

    class _Connection:  # placeholder for type hints
        pass

    class _Cursor:
        pass

    def _connect(*_a, **_k):
        raise _Error("pyodbc stub: no SQL driver in this environment")

    _stub.Error = _Error
    _stub.Connection = _Connection
    _stub.Cursor = _Cursor
    _stub.connect = _connect
    sys.modules["pyodbc"] = _stub

# --- 3. a Fernet key so security.encrypt_value works in tests ----------------
from cryptography.fernet import Fernet  # noqa: E402
os.environ.setdefault("FERNET_KEY", Fernet.generate_key().decode())

import pytest  # noqa: E402


# =============================================================================
# In-memory mock database
# =============================================================================
class MockDB:
    """A tiny in-memory stand-in for the Anima schema (the 'mock dataset')."""

    def __init__(self):
        self.patients = {}       # patient_ID -> {name, biological_sex, is_child, age}
        self.assessments = []    # [{assessment_ID, patient_ID, ...}]
        self.mri = []            # [{patient_ID, scan_type, file_path, is_current, scan_session, slice_count}]
        self._aid = 0

    # -- assessment helpers --
    def add_assessment(self, patient_id, **fields):
        self._aid += 1
        row = {"assessment_ID": self._aid, "patient_ID": patient_id,
               "raw_answers": None, "clinician_notes": None}
        row.update(fields)
        self.assessments.append(row)
        return self._aid

    def latest_aid(self, patient_id):
        aids = [r["assessment_ID"] for r in self.assessments if r["patient_ID"] == patient_id]
        return max(aids) if aids else None

    def update_assessment(self, aid, **fields):
        for r in self.assessments:
            if r["assessment_ID"] == aid:
                r.update(fields)

    def assessment_for(self, patient_id):
        rows = [r for r in self.assessments if r["patient_ID"] == patient_id]
        return rows[-1] if rows else None

    # -- MRI helpers --
    def max_session(self, patient_id, current_only=False):
        s = [r["scan_session"] for r in self.mri
             if r["patient_ID"] == patient_id and (not current_only or r["is_current"] == 1)]
        return max(s) if s else None

    def current_mri(self, patient_id=None):
        rows = [r for r in self.mri if r["is_current"] == 1]
        return [r for r in rows if patient_id is None or r["patient_ID"] == patient_id]


class FakeCursor:
    """Emulates the exact SQL the ingestion code runs against MockDB."""

    def __init__(self, db):
        self.db = db
        self._last = []

    def execute(self, sql, *params):
        q = " ".join(str(sql).split())
        db = self.db
        self._last = []

        if q.startswith("SELECT 1 FROM dbo.Patient WHERE patient_ID"):
            self._last = [(1,)] if params[0] in db.patients else []

        elif q.startswith("INSERT INTO dbo.Patient"):
            pid, name, sex, is_child, age = params
            db.patients[pid] = {"name": name, "biological_sex": sex,
                                "is_child": is_child, "age": age}

        elif q.startswith("INSERT INTO dbo.DSM5_Assessment (patient_ID, ground_truth_dx"):
            pid, dx, adhd_index, inatt, hyper, iq, med = params
            db.add_assessment(pid, ground_truth_dx=dx, adhd_index=adhd_index,
                              inattentive_score=inatt, hyperactive_score=hyper,
                              iq_measure=iq, med_status=med)

        elif q.startswith("INSERT INTO dbo.DSM5_Assessment (patient_ID, raw_answers"):
            pid, raw, notes = params
            db.add_assessment(pid, raw_answers=raw, clinician_notes=notes)

        elif q.startswith("SELECT TOP 1 assessment_ID FROM dbo.DSM5_Assessment"):
            aid = db.latest_aid(params[0])
            self._last = [(aid,)] if aid is not None else []

        # DSM5_Analysis.analyze_patient: latest assessment joined with demographics.
        elif q.startswith("SELECT TOP 1 p.age, p.biological_sex, p.is_child, d.assessment_ID"):
            a = db.assessment_for(params[0])
            p = db.patients.get(params[0])
            if a is None or p is None:
                self._last = []
            else:
                self._last = [(p.get("age"), p.get("biological_sex"), p.get("is_child"),
                               a["assessment_ID"], a.get("inattentive_score"),
                               a.get("hyperactive_score"), a.get("clinician_notes"))]

        elif q.startswith("UPDATE dbo.DSM5_Assessment SET nlp_risk_score"):
            risk, aid = params
            db.update_assessment(aid, nlp_risk_score=risk)

        elif q.startswith("UPDATE dbo.DSM5_Assessment SET raw_answers"):
            raw, notes, aid = params
            db.update_assessment(aid, raw_answers=raw, clinician_notes=notes)

        # ---- MRI ----
        elif q.startswith("SELECT MAX(scan_session) FROM dbo.MRI") and "is_current" not in q:
            self._last = [(db.max_session(params[0], current_only=False),)]

        elif "SELECT MAX(scan_session) FROM dbo.MRI" in q and "is_current = 1" in q:
            self._last = [(db.max_session(params[0], current_only=True),)]

        elif q.startswith("SELECT file_path FROM dbo.MRI"):
            pid, scan_type = params
            self._last = [(r["file_path"],) for r in db.mri
                          if r["patient_ID"] == pid and r["scan_type"] == scan_type
                          and r["is_current"] == 1]

        elif q.startswith("DELETE FROM dbo.MRI"):
            pid, scan_type = params
            db.mri = [r for r in db.mri if not (r["patient_ID"] == pid
                      and r["scan_type"] == scan_type and r["is_current"] == 1)]

        elif q.startswith("UPDATE dbo.MRI SET is_current = 0"):
            pid, scan_type = params
            for r in db.mri:
                if r["patient_ID"] == pid and r["scan_type"] == scan_type and r["is_current"] == 1:
                    r["is_current"] = 0

        elif q.startswith("INSERT INTO dbo.MRI"):
            pid, scan_type, file_path, slice_count, session = params
            db.mri.append({"patient_ID": pid, "scan_type": scan_type,
                           "file_path": file_path, "slice_count": slice_count,
                           "is_current": 1, "scan_session": session})

        return self

    def fetchone(self):
        return self._last[0] if self._last else None

    def fetchall(self):
        return list(self._last)


class FakeConn:
    def __init__(self, db):
        self.db = db
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return FakeCursor(self.db)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


def pytest_collection_modifyitems(config, items):
    """Don't run `live` (real-DB, write) tests unless explicitly selected with
    `-m live`. Keeps a plain `pytest` fast, offline, and side-effect free."""
    markexpr = config.option.markexpr or ""
    if "live" in markexpr:
        return   # user asked for the live tests on purpose
    skip_live = pytest.mark.skip(reason="live DB test - run with: pytest -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def mockdb():
    """A fresh in-memory mock dataset per test."""
    return MockDB()


@pytest.fixture
def fake_conn(mockdb):
    """A fake DB connection bound to the per-test mock dataset."""
    return FakeConn(mockdb)
