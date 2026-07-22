"""
LIVE integration test: seed patients 130-222 into the real database.

Runs the actual ``seed_dsm5.py`` script against your running SQL Server for the
130-222 CSVs, then verifies the CORRECT tables were updated:

  * the new patients land in Patient AND DSM5_Assessment (in lock-step);
  * clinician notes are attached;
  * MRI and Analysis_Result are left untouched;
  * a sample of patients has the right demographics / DX in the right columns;
  * re-running is idempotent (no duplicates).

Marked ``live`` and auto-skipped if SQL Server is unreachable or the CSVs are
missing, so it never breaks the offline unit run. It DOES write to the database
(that's the point) - it is idempotent, so safe to re-run.

Run it explicitly (DB up, .env configured):
    pytest tests/ingestion -m live
"""

import os
import sys
import subprocess

import pandas as pd
import pytest

from database import get_connection
from CSV_Ingestion import _clean, _to_int

pytestmark = pytest.mark.live

# Repo layout: tests/ingestion/this_file -> repo root is two levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
_DATA = os.path.join(_REPO_ROOT, "data")
PHENO_CSV = os.path.join(_DATA, "NYU_Athena_Phenotypic_130-222.csv")
DSM5_CSV = os.path.join(_DATA, "DSM5_data_130-222.csv")

TABLES = ("Patient", "DSM5_Assessment", "MRI", "Analysis_Result")


def _counts(conn) -> dict:
    cur = conn.cursor()
    out = {}
    for t in TABLES:
        cur.execute(f"SELECT COUNT(*) FROM dbo.{t};")
        out[t] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM dbo.DSM5_Assessment WHERE clinician_notes IS NOT NULL;")
    out["notes"] = cur.fetchone()[0]
    return out


def _expected_ids():
    """The 7-digit patient IDs the seeder will derive from the 130-222 CSV."""
    df = pd.read_csv(PHENO_CSV, dtype=str, keep_default_na=False)
    ids = []
    for _, row in df.iterrows():
        raw = _clean(row["ScanDir ID"])
        if raw:
            ids.append(raw.split(".")[0].zfill(7))
    return ids


@pytest.fixture(scope="module")
def live_conn():
    """A live connection, or skip the whole module if SQL Server isn't reachable."""
    if not (os.path.exists(PHENO_CSV) and os.path.exists(DSM5_CSV)):
        pytest.skip(f"130-222 CSVs not found in {_DATA}")
    try:
        conn = get_connection()
        conn.cursor().execute("SELECT 1;")
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"live SQL Server not reachable: {exc}")
    yield conn
    conn.close()


def _run_seed():
    """Run the real seed_dsm5.py for the 130-222 files (schema assumed to exist)."""
    return subprocess.run(
        [sys.executable, "seed_dsm5.py", "--skip-schema",
         "--phenotypic", PHENO_CSV, "--dsm5", DSM5_CSV],
        cwd=_APP_DIR, capture_output=True, text=True, timeout=600,
    )


@pytest.fixture(scope="module")
def seeded(live_conn):
    """Seed 130-222 once, capturing table counts before and after.

    Also records how many of the expected patients were NOT already in the DB, so
    the delta assertions hold whether the DB is fresh (93 inserted) or already
    seeded from a previous run (0 inserted - the seeder is idempotent)."""
    cur = live_conn.cursor()
    cur.execute("SELECT patient_ID FROM dbo.Patient;")
    present = {str(r[0]) for r in cur.fetchall()}
    expected_new = sum(1 for pid in _expected_ids() if pid not in present)

    before = _counts(live_conn)
    proc = _run_seed()
    after = _counts(live_conn)
    return {"proc": proc, "before": before, "after": after, "expected_new": expected_new}


# ---------------------------------------------------------------------------
def test_seed_script_succeeds(seeded):
    proc = seeded["proc"]
    assert proc.returncode == 0, f"seeder failed:\n{proc.stdout}\n{proc.stderr}"
    assert "Seed complete" in proc.stdout


def test_new_patients_added_to_patient_and_assessment(live_conn, seeded):
    ids = _expected_ids()
    cur = live_conn.cursor()
    # every expected patient exists and has an assessment
    for pid in ids:
        cur.execute("SELECT 1 FROM dbo.Patient WHERE patient_ID = ?;", pid)
        assert cur.fetchone() is not None, f"patient {pid} missing from Patient"
        cur.execute("SELECT COUNT(*) FROM dbo.DSM5_Assessment WHERE patient_ID = ?;", pid)
        assert cur.fetchone()[0] >= 1, f"patient {pid} has no DSM5_Assessment"


def test_only_the_correct_tables_changed(seeded):
    before, after, expected_new = (
        seeded["before"], seeded["after"], seeded["expected_new"])
    added_patients = after["Patient"] - before["Patient"]
    added_assessments = after["DSM5_Assessment"] - before["DSM5_Assessment"]
    # Exactly the not-yet-present 130-222 patients were inserted: 93 on a fresh DB,
    # 0 if they were already seeded (idempotent) - either way Patient and
    # DSM5_Assessment move in lock-step (one assessment per new patient).
    assert added_patients == expected_new
    assert added_assessments == expected_new
    # Notes never decrease; demographic ingestion must NOT touch imaging / results.
    assert after["notes"] >= before["notes"]
    assert after["MRI"] == before["MRI"]
    assert after["Analysis_Result"] == before["Analysis_Result"]


def test_sample_patient_data_is_correct(live_conn, seeded):
    """A few patients have the right demographics + DX in the right columns."""
    df = pd.read_csv(PHENO_CSV, dtype=str, keep_default_na=False)
    checked = 0
    cur = live_conn.cursor()
    for _, row in df.iterrows():
        raw = _clean(row["ScanDir ID"])
        dx = _to_int(row["DX"])
        age = _clean(row["Age"])
        if not raw or dx is None or age is None:
            continue
        pid = raw.split(".")[0].zfill(7)
        cur.execute("SELECT age, biological_sex, is_child FROM dbo.Patient WHERE patient_ID = ?;", pid)
        prow = cur.fetchone()
        assert prow is not None
        db_age, db_sex, db_child = prow
        assert abs(float(db_age) - float(age)) < 0.01
        assert int(db_sex) == _to_int(row["Gender"])
        assert int(db_child) == (1 if float(age) < 18 else 0)
        cur.execute(
            "SELECT TOP 1 ground_truth_dx FROM dbo.DSM5_Assessment "
            "WHERE patient_ID = ? ORDER BY assessment_ID DESC;", pid)
        assert cur.fetchone()[0] == dx
        checked += 1
        if checked >= 3:
            break
    assert checked >= 1, "no sample rows with clean DX/Age to verify"


def test_reseeding_is_idempotent(live_conn, seeded):
    """Running the seeder a second time inserts no duplicates."""
    before = _counts(live_conn)
    proc = _run_seed()
    after = _counts(live_conn)
    assert proc.returncode == 0
    assert after["Patient"] == before["Patient"]           # all skipped
    assert after["DSM5_Assessment"] == before["DSM5_Assessment"]
