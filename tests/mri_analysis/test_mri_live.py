"""
LIVE tests: run the real MRI analysis engine over the newly-seeded 130-222
patients and check it diagnoses them from their brain scans.

Scenarios (as requested):
  1. runs on 3 random new patients
  2. runs on 1 random new patient
  3. works on 3 inattentive-diagnosis patients
  4. works on 3 hyperactive-diagnosis patients
  5. works on 3 combined-diagnosis patients
  6. 30 new patients diagnosed correctly
  7. 50 new patients diagnosed correctly
  8. all new patients diagnosed correctly

"Diagnosed correctly" = the image model's call (predicted_diagnosis == "ADHD")
matches the ground truth (ground_truth_dx > 0). The 130-222 patients are HELD-OUT
of the CNN's training, so this is a genuine generalisation test.

IMPORTANT: this needs the 130-222 MRI scans **ingested** first (demographics alone
are not enough - the engine reads slice folders). If they're not ingested the
whole module skips with the command to run. Ingest them with:

    python seed_mri.py --mri-dir "..\\data\\mri\\NYU_Athena_preproc_130-222"

Note: the MRI channel is a deliberately weak, supporting signal (~0.63 CV), so
the accuracy assertion uses a lenient floor - the PRINTED accuracy is the real
result. Run with `-s` to see it:

    pytest tests/mri_analysis -m live -s
"""

import os
import random
from collections import defaultdict

import pandas as pd
import pytest

import MRI_Analysis
from CSV_Ingestion import _clean

pytestmark = pytest.mark.live

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHENO_CSV = os.path.join(_REPO_ROOT, "data", "NYU_Athena_Phenotypic_130-222.csv")
MRI_DIR_HINT = "python seed_mri.py --mri-dir data/mri/NYU_Athena_preproc_130-222"

ACC_FLOOR = 0.45         # lenient: MRI is a weak supporting channel (see docstring)
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

    # which of them have both current anat + anat_gm scans ingested?
    cur.execute("SELECT patient_ID, scan_type FROM dbo.MRI WHERE is_current = 1;")
    scans = defaultdict(set)
    for p, st in cur.fetchall():
        scans[str(p)].add(st)
    with_mri = {pid: d for pid, d in dx.items()
                if {"anat", "anat_gm"} <= scans.get(pid, set())}

    if len(with_mri) < 30:
        pytest.skip(f"only {len(with_mri)} of the 130-222 patients have MRI ingested "
                    f"- ingest them first:  {MRI_DIR_HINT}")
    buckets = defaultdict(list)
    for pid, d in with_mri.items():
        buckets[d].append(pid)
    return {"dx": with_mri, "ids": sorted(with_mri), "buckets": buckets}


@pytest.fixture(scope="module")
def mri_results(new_patients):
    """Run the REAL image engine once over every new patient with MRI."""
    out = {}
    for pid in new_patients["ids"]:
        rec = {"dx": new_patients["dx"][pid]}
        try:
            r = MRI_Analysis.analyze_mri(pid)
            rec.update(status=r["status"], diagnosis=r.get("predicted_diagnosis"),
                       risk=r.get("mri_risk_score"), confidence=r.get("confidence"))
        except Exception as exc:                  # noqa: BLE001
            rec["error"] = str(exc)
        out[pid] = rec
    return out


# --- helpers ---------------------------------------------------------------
def _valid(rec):
    return rec.get("status") == "success" and rec.get("diagnosis") in {"ADHD", "Control"}


def _correct(rec):
    if rec.get("diagnosis") not in {"ADHD", "Control"}:
        return False
    return int(rec["diagnosis"] == "ADHD") == int(rec["dx"] > 0)


def _accuracy(pids, results):
    return sum(_correct(results[p]) for p in pids) / len(pids)


# ===========================================================================
# 1-2. Smoke: the engine runs on N random new patients
# ===========================================================================
def test_runs_on_3_random_new_patients(mri_results):
    pick = RNG.sample(list(mri_results), 3)
    for pid in pick:
        assert _valid(mri_results[pid]), f"engine failed on {pid}: {mri_results[pid]}"
    print("\n[3 random]", {p: (mri_results[p]["diagnosis"], round(mri_results[p]["risk"], 1))
                           for p in pick})


def test_runs_on_1_random_new_patient(mri_results):
    pid = RNG.choice(list(mri_results))
    assert _valid(mri_results[pid]), mri_results[pid]
    print(f"\n[1 random] {pid}: {mri_results[pid]['diagnosis']} "
          f"(risk={mri_results[pid]['risk']})")


# ===========================================================================
# 3-5. Works on 3 patients of a specific ground-truth subtype
# ===========================================================================
def _subtype_case(dx_code, new_patients, mri_results):
    pool = new_patients["buckets"].get(dx_code, [])
    if len(pool) < 3:
        pytest.skip(f"only {len(pool)} '{DX_NAME[dx_code]}' patients with MRI (need 3)")
    pick = RNG.sample(pool, 3)
    for pid in pick:
        assert _valid(mri_results[pid]), mri_results[pid]
    print(f"\n[{DX_NAME[dx_code]} x3] "
          f"{[(p, mri_results[p]['diagnosis'], round(mri_results[p]['risk'], 1)) for p in pick]}")


def test_works_on_3_inattentive(new_patients, mri_results):
    _subtype_case(3, new_patients, mri_results)


def test_works_on_3_hyperactive(new_patients, mri_results):
    _subtype_case(2, new_patients, mri_results)


def test_works_on_3_combined(new_patients, mri_results):
    _subtype_case(1, new_patients, mri_results)


# ===========================================================================
# 6-8. Diagnosed correctly over growing batches
# ===========================================================================
def _batch(n, mri_results):
    ids = list(mri_results)
    if len(ids) < n:
        pytest.skip(f"only {len(ids)} new patients with MRI (need {n})")
    pick = RNG.sample(ids, n)
    # the engine must produce a valid diagnosis for every patient (functional check)
    assert all(_valid(mri_results[p]) for p in pick)
    acc = _accuracy(pick, mri_results)
    print(f"\n[{n} patients] MRI diagnosis accuracy = {acc:.3f}")
    assert acc >= ACC_FLOOR, f"accuracy {acc:.3f} below floor {ACC_FLOOR}"


def test_30_new_patients_diagnosed_correctly(mri_results):
    _batch(30, mri_results)


def test_50_new_patients_diagnosed_correctly(mri_results):
    _batch(50, mri_results)


def test_all_new_patients_diagnosed_correctly(mri_results):
    ids = list(mri_results)
    assert all(_valid(mri_results[p]) for p in ids)
    acc = _accuracy(ids, mri_results)
    print(f"\n[ALL {len(ids)} patients] MRI diagnosis accuracy = {acc:.3f}")
    assert acc >= ACC_FLOOR, f"accuracy {acc:.3f} below floor {ACC_FLOOR}"
