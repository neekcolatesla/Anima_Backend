"""
Unit tests for CSV ingestion (app/CSV_Ingestion.py).

Exercises the REAL ingestion endpoints against the in-memory mock dataset
(conftest.MockDB) via FastAPI's TestClient with the DB dependency overridden - so
we feed mock CSV data in and assert exactly what lands in the mock Patient /
DSM5_Assessment tables. No SQL Server needed.
"""

import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import CSV_Ingestion
from CSV_Ingestion import _clean, _to_int, _to_decimal
from database import get_db
from security import decrypt_value

pytestmark = pytest.mark.unit

# Header the phenotypic endpoint requires.
_PHENO_HEADER = ("ScanDir ID,Age,Gender,Name,DX,ADHD Index,Inattentive,"
                 "Hyper/Impulsive,IQ Measure,Med Status")


def _csv(rows, header=_PHENO_HEADER):
    return (header + "\n" + "\n".join(rows) + "\n").encode("utf-8")


# A small mock phenotypic dataset: an adult, a child (with -999/N/A nulls), and a
# decimal ScanDir ID to check 7-digit normalisation.
_PHENO_ROWS = [
    "10130,25,1,Alice Adult,1,65,70,60,110,1",
    "10131,12,0,Bob Child,0,-999,N/A,45,100,0",
    "10132.0,30,1,Carol Adult,3,72,68,71,105,1",
]


@pytest.fixture
def client(fake_conn):
    """A bare app with just the ingestion router, DB overridden to the mock."""
    app = FastAPI()
    app.include_router(CSV_Ingestion.router)
    app.dependency_overrides[get_db] = lambda: fake_conn
    return TestClient(app), fake_conn.db


def _upload(client, path, data, name="data.csv"):
    return client.post(path, files={"file": (name, io.BytesIO(data), "text/csv")})


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def test_clean_maps_sentinels_to_none():
    assert _clean("-999") is None
    assert _clean("N/A") is None
    assert _clean("  ") is None
    assert _clean("  Dana  ") == "Dana"


def test_numeric_coercion():
    assert _to_int("3") == 3
    assert _to_int("3.0") == 3        # tolerated float-like
    assert _to_int("-999") is None
    assert _to_decimal("12.5") == 12.5
    assert _to_decimal("N/A") is None


# ---------------------------------------------------------------------------
# Phenotypic CSV ingestion into the mock dataset
# ---------------------------------------------------------------------------
def test_ingest_adds_patients_to_mock_dataset(client):
    tc, db = client
    r = _upload(tc, "/api/ingest/csv", _csv(_PHENO_ROWS))
    assert r.status_code == 200
    body = r.json()
    assert body["ingested_count"] == 3 and body["skipped_existing"] == 0
    # 7-digit IDs (leading zeros preserved; decimal stripped)
    assert set(db.patients) == {"0010130", "0010131", "0010132"}
    assert len(db.assessments) == 3


def test_child_flag_and_null_scores(client):
    tc, db = client
    _upload(tc, "/api/ingest/csv", _csv(_PHENO_ROWS))
    assert db.patients["0010131"]["is_child"] == 1     # age 12 -> child
    assert db.patients["0010130"]["is_child"] == 0     # age 25 -> adult
    child = db.assessment_for("0010131")
    assert child["adhd_index"] is None                 # -999 -> NULL
    assert child["inattentive_score"] is None          # N/A -> NULL


def test_name_encrypted_at_rest(client):
    tc, db = client
    _upload(tc, "/api/ingest/csv", _csv(_PHENO_ROWS))
    stored = db.patients["0010130"]["name"]
    assert stored != "Alice Adult"                     # not plaintext
    assert decrypt_value(stored) == "Alice Adult"      # round-trips


def test_dx_and_scores_mapped(client):
    tc, db = client
    _upload(tc, "/api/ingest/csv", _csv(_PHENO_ROWS))
    a = db.assessment_for("0010132")
    assert a["ground_truth_dx"] == 3
    assert a["inattentive_score"] == 68 and a["hyperactive_score"] == 71


def test_ingest_is_idempotent(client):
    tc, db = client
    _upload(tc, "/api/ingest/csv", _csv(_PHENO_ROWS))
    r2 = _upload(tc, "/api/ingest/csv", _csv(_PHENO_ROWS))
    body = r2.json()
    assert body["ingested_count"] == 0 and body["skipped_existing"] == 3
    assert len(db.patients) == 3                       # no duplicates


def test_missing_required_column_rejected(client):
    tc, _ = client
    bad = _csv(["10130,25,1,Alice,65,70,60,110,1"],
               header="ScanDir ID,Age,Gender,Name,ADHD Index,Inattentive,"
                      "Hyper/Impulsive,IQ Measure,Med Status")   # no DX column
    r = _upload(tc, "/api/ingest/csv", bad)
    assert r.status_code == 400 and "DX" in r.json()["detail"]


def test_empty_file_rejected(client):
    tc, _ = client
    r = _upload(tc, "/api/ingest/csv", b"")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# DSM-5 CSV enrichment (notes + answers onto the latest assessment)
# ---------------------------------------------------------------------------
def test_dsm5_csv_enriches_latest_assessment(client):
    tc, db = client
    _upload(tc, "/api/ingest/csv", _csv(_PHENO_ROWS))         # create assessments first
    dsm5 = ('patient_ID,raw_answers_json,clinician_notes\n'
            '0010130,"{""q1"": 2}",Patient reports restlessness.\n'
            '9999999,"{""q1"": 1}",Ghost patient.\n'                 # missing patient -> skipped
            '0010131,not-json,Bad answers.\n')                       # bad JSON -> skipped
    r = _upload(tc, "/api/ingest/dsm5-csv", dsm5.encode("utf-8"))
    assert r.status_code == 200
    body = r.json()
    assert body["updated_count"] == 1
    assert body["skipped_missing_patient"] == 1
    assert body["skipped_invalid_json"] == 1
    a = db.assessment_for("0010130")
    assert a["clinician_notes"] == "Patient reports restlessness."
    assert a["raw_answers"] == '{"q1": 2}'
