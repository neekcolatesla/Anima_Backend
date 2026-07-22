"""
LIVE tests: run the real Combined analysis engine over the newly-seeded 130-222
patients (both channels available: DSM-5 seeded + MRI ingested).

Scenarios (as requested): 3 random, 1 random, 3 inattentive, 3 hyperactive,
3 combined, then 30 / 50 / all diagnosed correctly.

"Diagnosed correctly" = the fused call (predicted_diagnosis == "ADHD") matches the
ground truth (ground_truth_dx > 0). The fusion is NLP-weighted 0.7 / MRI 0.3, so
the combined accuracy should track the strong NLP channel; the all-patients test
also prints the NLP-only and MRI-only accuracies for comparison.

Needs the DB up, .env set, 130-222 DSM-5 seeded AND MRI ingested (see the MRI live
test). Auto-skips otherwise. Writes one Analysis_Result audit row per patient (the
engine's normal behaviour). Slow: runs both models per patient (a few minutes).
    pytest tests/combined_analysis -m live -s
"""

import os
import random
from collections import defaultdict

import pandas as pd
import pytest

import Combined_Analysis
from CSV_Ingestion import _clean

pytestmark = pytest.mark.live

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHENO_CSV = os.path.join(_REPO_ROOT, "data", "NYU_Athena_Phenotypic_130-222.csv")

TRIGGERED_BY = "A000001"     # the seeded admin (created_by on the audit rows)
ACC_FLOOR = 0.60             # combined is NLP-dominant, so a real floor applies
DX_NAME = {1: "Combined", 2: "Hyperactive", 3: "Inattentive"}
RNG = random.Random(20260721)


def _expected_ids():
    df = pd.read_csv(PHENO_CSV, dtype=str, keep_default_na=False)
    return {_clean(r["ScanDir ID"]).split(".")[0].zfill(7)
            for _, r in df.iterrows() if _clean(r["ScanDir ID"])}


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
    """130-222 patients that have a diagnosis AND both current MRI scans."""
    ids = _expected_ids()
    cur = live_conn.cursor()
    cur.execute("SELECT patient_ID, ground_truth_dx FROM dbo.DSM5_Assessment "
                "WHERE ground_truth_dx IS NOT NULL;")
    dx = {str(p): int(d) for p, d in cur.fetchall() if str(p) in ids}
    cur.execute("SELECT patient_ID, scan_type FROM dbo.MRI WHERE is_current = 1;")
    scans = defaultdict(set)
    for p, st in cur.fetchall():
        scans[str(p)].add(st)
    usable = {pid: d for pid, d in dx.items()
              if {"anat", "anat_gm"} <= scans.get(pid, set())}
    if len(usable) < 30:
        pytest.skip(f"only {len(usable)} of 130-222 have BOTH demographics and MRI - "
                    f"seed DSM-5 (pytest tests/ingestion -m live) and ingest MRI "
                    f"(python seed_mri.py --mri-dir data/mri/NYU_Athena_preproc_130-222)")
    buckets = defaultdict(list)
    for pid, d in usable.items():
        buckets[d].append(pid)
    return {"dx": usable, "ids": sorted(usable), "buckets": buckets}


@pytest.fixture(scope="module")
def combined_results(new_patients):
    """Run the REAL combined engine once over every usable new patient."""
    out = {}
    for pid in new_patients["ids"]:
        rec = {"dx": new_patients["dx"][pid]}
        try:
            r = Combined_Analysis.analyze_combined(pid, created_by=TRIGGERED_BY)
            rec.update(status=r["status"], diagnosis=r["predicted_diagnosis"],
                       combined=r["final_combined_score"], nlp=r["nlp_risk_score"],
                       mri=r["mri_risk_score"], mri_status=r["mri_status"],
                       subtype=r["predicted_subtype"])
        except Exception as exc:                  # noqa: BLE001
            rec["error"] = str(exc)
        out[pid] = rec
    return out


# --- helpers ---------------------------------------------------------------
def _valid(rec):
    return (rec.get("status") == "success"
            and rec.get("diagnosis") in {"ADHD", "Control"}
            and isinstance(rec.get("combined"), (int, float)))


def _correct(rec):
    if rec.get("diagnosis") not in {"ADHD", "Control"}:
        return False
    return int(rec["diagnosis"] == "ADHD") == int(rec["dx"] > 0)


def _acc(pids, results):
    return sum(_correct(results[p]) for p in pids) / len(pids)


# ===========================================================================
# 1-2. Smoke
# ===========================================================================
def test_runs_on_3_random_new_patients(combined_results):
    pick = RNG.sample(list(combined_results), 3)
    for pid in pick:
        rec = combined_results[pid]
        assert _valid(rec), f"engine failed on {pid}: {rec}"
        assert rec["mri_status"] == "success"    # genuine fusion, not NLP-only fallback
    print("\n[3 random]", {p: (combined_results[p]["diagnosis"],
                               round(combined_results[p]["combined"], 1)) for p in pick})


def test_runs_on_1_random_new_patient(combined_results):
    pid = RNG.choice(list(combined_results))
    rec = combined_results[pid]
    assert _valid(rec), rec
    print(f"\n[1 random] {pid}: {rec['diagnosis']} combined={rec['combined']} "
          f"(nlp={rec['nlp']}, mri={rec['mri']})")


# ===========================================================================
# 3-5. Works on 3 patients of a specific ground-truth subtype
# ===========================================================================
def _subtype_case(dx_code, new_patients, combined_results):
    pool = new_patients["buckets"].get(dx_code, [])
    if len(pool) < 3:
        pytest.skip(f"only {len(pool)} '{DX_NAME[dx_code]}' patients (need 3)")
    pick = RNG.sample(pool, 3)
    for pid in pick:
        assert _valid(combined_results[pid]), combined_results[pid]
    flagged = sum(combined_results[p]["diagnosis"] == "ADHD" for p in pick)
    print(f"\n[{DX_NAME[dx_code]} x3] "
          f"{[(p, combined_results[p]['diagnosis'], combined_results[p]['subtype']) for p in pick]} "
          f"flagged ADHD={flagged}/3")
    assert flagged >= 2                          # ADHD patients -> mostly flagged (NLP-driven)


def test_works_on_3_inattentive(new_patients, combined_results):
    _subtype_case(3, new_patients, combined_results)


def test_works_on_3_hyperactive(new_patients, combined_results):
    _subtype_case(2, new_patients, combined_results)


def test_works_on_3_combined(new_patients, combined_results):
    _subtype_case(1, new_patients, combined_results)


# ===========================================================================
# 6-8. Diagnosed correctly over growing batches
# ===========================================================================
def _batch(n, combined_results):
    ids = list(combined_results)
    if len(ids) < n:
        pytest.skip(f"only {len(ids)} usable new patients (need {n})")
    pick = RNG.sample(ids, n)
    assert all(_valid(combined_results[p]) for p in pick)
    acc = _acc(pick, combined_results)
    print(f"\n[{n} patients] combined diagnosis accuracy = {acc:.3f}")
    assert acc >= ACC_FLOOR, f"accuracy {acc:.3f} below floor {ACC_FLOOR}"


def test_30_new_patients_diagnosed_correctly(combined_results):
    _batch(30, combined_results)


def test_50_new_patients_diagnosed_correctly(combined_results):
    _batch(50, combined_results)


def test_all_new_patients_diagnosed_correctly(combined_results):
    ids = list(combined_results)
    assert all(_valid(combined_results[p]) for p in ids)
    acc = _acc(ids, combined_results)
    # component accuracies for comparison (shows the fusion effect)
    nlp_acc = sum(int(combined_results[p]["nlp"] >= 50) == int(combined_results[p]["dx"] > 0)
                  for p in ids) / len(ids)
    mri_acc = sum(int((combined_results[p]["mri"] or 0) >= 50) == int(combined_results[p]["dx"] > 0)
                  for p in ids) / len(ids)
    print(f"\n[ALL {len(ids)} patients] combined={acc:.3f}  "
          f"(nlp-only={nlp_acc:.3f}, mri-only={mri_acc:.3f})")
    assert acc >= ACC_FLOOR, f"accuracy {acc:.3f} below floor {ACC_FLOOR}"
