"""
LIVE tests: run the real DSM-5 (NLP) analysis engine over the newly-seeded
130-222 patients and check it diagnoses them.

Scenarios (as requested):
  1. runs on 3 random new patients
  2. runs on 1 random new patient
  3. works on 3 inattentive-diagnosis patients
  4. works on 3 hyperactive-diagnosis patients
  5. works on 3 combined-diagnosis patients
  6. 30 new patients diagnosed correctly
  7. 50 new patients diagnosed correctly
  8. all 93 new patients diagnosed correctly

"Diagnosed correctly" = the engine's ADHD-vs-control call (nlp_risk_score >=
NLP_THRESHOLD) matches the ground-truth diagnosis (ground_truth_dx > 0). The
actual accuracy is printed (run with `-s`); the assertion only checks a
sensible floor, since these 130-222 patients are HELD-OUT (the model never saw
them), so this is a genuine generalisation test.

Marked `live`: needs the DB up, .env set, and 130-222 already seeded
(`pytest tests/ingestion -m live`). Auto-skips otherwise. Runs the engine once
over every new patient (a module fixture); the scenarios slice that result set.
Run it explicitly:
    pytest tests/dsm5_analysis -m live -s
"""

import os
import random
from collections import defaultdict

import pandas as pd
import pytest

import DSM5_Analysis
from CSV_Ingestion import _clean

pytestmark = pytest.mark.live

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHENO_CSV = os.path.join(_REPO_ROOT, "data", "NYU_Athena_Phenotypic_130-222.csv")

NLP_THRESHOLD = 50.0     # nlp_risk_score at/above which the engine calls "ADHD"
ACC_FLOOR = 0.60         # conservative "clearly better than chance" floor
DX_NAME = {1: "Combined", 2: "Hyperactive", 3: "Inattentive"}
RNG = random.Random(20260721)   # fixed so "random" picks are reproducible


def _expected_ids():
    df = pd.read_csv(PHENO_CSV, dtype=str, keep_default_na=False)
    return [ _clean(r["ScanDir ID"]).split(".")[0].zfill(7)
             for _, r in df.iterrows() if _clean(r["ScanDir ID"]) ]


@pytest.fixture(scope="module")
def live_conn():
    if not os.path.exists(PHENO_CSV):
        pytest.skip("130-222 phenotypic CSV not found")
    try:
        from database import get_connection
        conn = get_connection()
        conn.cursor().execute("SELECT 1;")
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"live SQL Server not reachable: {exc}")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def new_patients(live_conn):
    """{patient_ID: ground_truth_dx} for the seeded 130-222 patients, + dx buckets."""
    ids = set(_expected_ids())
    cur = live_conn.cursor()
    cur.execute("SELECT patient_ID, ground_truth_dx FROM dbo.DSM5_Assessment "
                "WHERE ground_truth_dx IS NOT NULL;")
    dx = {str(p): int(d) for p, d in cur.fetchall() if str(p) in ids}
    if len(dx) < 30:
        pytest.skip(f"only {len(dx)} of the 130-222 patients are seeded - run "
                    f"`pytest tests/ingestion -m live` first")
    buckets = defaultdict(list)
    for pid, d in dx.items():
        buckets[d].append(pid)
    return {"dx": dx, "ids": sorted(dx), "buckets": buckets}


@pytest.fixture(scope="module")
def nlp_results(new_patients):
    """Run the REAL engine once over every new patient. {pid: {dx, risk, ...}}."""
    out = {}
    for pid in new_patients["ids"]:
        rec = {"dx": new_patients["dx"][pid]}
        try:
            r = DSM5_Analysis.analyze_patient(pid)
            rec.update(risk=r["nlp_risk_score"], subtype=r["predicted_subtype"],
                       method=r["scoring_method"], status=r["status"])
        except Exception as exc:                  # noqa: BLE001
            rec["error"] = str(exc)
        out[pid] = rec
    return out


# --- helpers ---------------------------------------------------------------
def _valid(rec):
    return (rec.get("status") == "success" and isinstance(rec.get("risk"), (int, float))
            and 0 <= rec["risk"] <= 100
            and rec.get("subtype") in {"Combined", "Inattentive", "Hyperactive", "None"})


def _correct(rec):
    if "risk" not in rec:
        return False
    return int(rec["risk"] >= NLP_THRESHOLD) == int(rec["dx"] > 0)


def _accuracy(pids, results):
    return sum(_correct(results[p]) for p in pids) / len(pids)


# ===========================================================================
# 1-2. Smoke: the engine runs on N random new patients
# ===========================================================================
def test_runs_on_3_random_new_patients(nlp_results):
    pick = RNG.sample(list(nlp_results), 3)
    for pid in pick:
        assert _valid(nlp_results[pid]), f"engine failed on {pid}: {nlp_results[pid]}"
    print("\n[3 random]", {p: round(nlp_results[p]["risk"], 1) for p in pick})


def test_runs_on_1_random_new_patient(nlp_results):
    pid = RNG.choice(list(nlp_results))
    assert _valid(nlp_results[pid]), nlp_results[pid]
    print(f"\n[1 random] {pid}: risk={nlp_results[pid]['risk']}, "
          f"subtype={nlp_results[pid]['subtype']}")


# ===========================================================================
# 3-5. Works on 3 patients of a specific ground-truth subtype
# ===========================================================================
def _subtype_case(dx_code, new_patients, nlp_results):
    pool = new_patients["buckets"].get(dx_code, [])
    if len(pool) < 3:
        pytest.skip(f"only {len(pool)} '{DX_NAME[dx_code]}' patients in 130-222 "
                    f"(need 3)")
    pick = RNG.sample(pool, 3)
    for pid in pick:
        assert _valid(nlp_results[pid]), nlp_results[pid]
    # they are all ADHD patients -> the engine should flag the majority as ADHD
    flagged = sum(nlp_results[p]["risk"] >= NLP_THRESHOLD for p in pick)
    print(f"\n[{DX_NAME[dx_code]} x3] risks="
          f"{[round(nlp_results[p]['risk'], 1) for p in pick]}, flagged ADHD={flagged}/3")
    assert flagged >= 2


def test_works_on_3_inattentive(new_patients, nlp_results):
    _subtype_case(3, new_patients, nlp_results)


def test_works_on_3_hyperactive(new_patients, nlp_results):
    _subtype_case(2, new_patients, nlp_results)


def test_works_on_3_combined(new_patients, nlp_results):
    _subtype_case(1, new_patients, nlp_results)


# ===========================================================================
# 6-8. Diagnosed correctly over growing batches
# ===========================================================================
def _batch_accuracy_test(n, nlp_results):
    ids = list(nlp_results)
    if len(ids) < n:
        pytest.skip(f"only {len(ids)} new patients available (need {n})")
    pick = RNG.sample(ids, n)
    acc = _accuracy(pick, nlp_results)
    methods = defaultdict(int)
    for p in pick:
        methods[nlp_results[p].get("method", "error")] += 1
    print(f"\n[{n} patients] diagnosis accuracy = {acc:.3f}  (scoring: {dict(methods)})")
    assert acc >= ACC_FLOOR, f"accuracy {acc:.3f} below floor {ACC_FLOOR}"


def test_30_new_patients_diagnosed_correctly(nlp_results):
    _batch_accuracy_test(30, nlp_results)


def test_50_new_patients_diagnosed_correctly(nlp_results):
    _batch_accuracy_test(50, nlp_results)


def test_all_new_patients_diagnosed_correctly(nlp_results):
    ids = list(nlp_results)
    acc = _accuracy(ids, nlp_results)
    methods = defaultdict(int)
    for p in ids:
        methods[nlp_results[p].get("method", "error")] += 1
    print(f"\n[ALL {len(ids)} patients] diagnosis accuracy = {acc:.3f}  "
          f"(scoring: {dict(methods)})")
    assert acc >= ACC_FLOOR, f"accuracy {acc:.3f} below floor {ACC_FLOOR}"
