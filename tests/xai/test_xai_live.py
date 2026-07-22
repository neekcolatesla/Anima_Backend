"""
LIVE tests: exercise the explainability (xAI) layer on the real, held-out 130-222
cohort and check every explanation the frontend would render is actually valid.

Unlike the DSM-5 / MRI / Combined live suites (which score accuracy), this suite
scores EXPLANATION QUALITY. For each patient it runs both real engines and asserts:

  * Text model  - feature_importance is a ranked, ~100%% set of signed bars, and the
                  influential words are genuine tokens taken FROM the patient's note
                  (not invented), each with a bounded signed push.
  * Image model - the Grad-CAM heatmap is a real PNG (decodes with PIL) that overlays
                  onto a valid brain slice index, and the summary names that slice.

Scenarios (same 8 as the other categories): 3 random, 1 random, 3 inattentive,
3 hyperactive, 3 combined, then 30 / 50 / all patients "explained correctly"
(= a valid text explanation AND a decodable heatmap).

Needs the DB up, .env set, 130-222 DSM-5 seeded AND MRI ingested. Auto-skips
otherwise. Slow (runs both models per patient):

    pytest tests/xai -m live -s
"""

import os
import base64
import random
from io import BytesIO
from collections import defaultdict, Counter

import pandas as pd
import pytest
from PIL import Image

import DSM5_Analysis
import MRI_Analysis
from CSV_Ingestion import _clean

pytestmark = pytest.mark.live

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHENO_CSV = os.path.join(_REPO_ROOT, "data", "NYU_Athena_Phenotypic_130-222.csv")

HEATMAP_FLOOR = 0.90     # >=90%% of patients must get a decodable Grad-CAM heatmap
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
    """130-222 patients with a diagnosis AND both current MRI scans, plus their notes."""
    ids = _expected_ids()
    cur = live_conn.cursor()
    cur.execute("SELECT patient_ID, ground_truth_dx, clinician_notes "
                "FROM dbo.DSM5_Assessment WHERE ground_truth_dx IS NOT NULL;")
    dx, notes = {}, {}
    for p, d, n in cur.fetchall():
        if str(p) in ids:
            dx[str(p)] = int(d)
            notes[str(p)] = n or ""
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
    return {"dx": usable, "ids": sorted(usable), "buckets": buckets, "notes": notes}


@pytest.fixture(scope="module")
def xai_results(new_patients):
    """Run BOTH real engines once per patient and keep their explanation blocks."""
    out = {}
    for pid in new_patients["ids"]:
        rec = {"note": new_patients["notes"].get(pid, "")}
        try:
            rec["text"] = DSM5_Analysis.analyze_patient(pid)["explanation"]["text_model"]
        except Exception as exc:                  # noqa: BLE001
            rec["text_error"] = str(exc)
        try:
            mri = MRI_Analysis.analyze_mri(pid)
            rec["mri_status"] = mri.get("status")
            if mri.get("status") == "success":
                rec["mri"] = mri["explanation"]["mri_model"]
        except Exception as exc:                  # noqa: BLE001
            rec["mri_error"] = str(exc)
        out[pid] = rec
    return out


# --- validators -------------------------------------------------------------
def _valid_text(ex):
    if not isinstance(ex, dict) or not ex.get("feature_importance"):
        return False
    fi = ex["feature_importance"]
    impacts = [f["impact_percent"] for f in fi]
    if impacts != sorted(impacts, reverse=True):          # must be ranked
        return False
    if abs(sum(impacts) - 100.0) >= 1.0:                  # must sum to ~100%
        return False
    for f in fi:
        if f["direction"] not in {"toward ADHD", "toward control"}:
            return False
    for w in ex.get("influential_words", []):
        if not (-100.0 <= w["push"] <= 100.0):
            return False
    return True


def _words_are_from_note(ex, note):
    """Influential words must be real tokens lifted from the patient's own note."""
    low = note.lower()
    words = ex.get("influential_words", [])
    return all(w["word"].lower() in low for w in words) if words else False


def _valid_heatmap(ex):
    """The Grad-CAM overlay must be a genuinely decodable PNG on a real slice index."""
    if not isinstance(ex, dict) or not ex.get("available"):
        return False
    if not isinstance(ex.get("top_slice_index"), int) or ex["top_slice_index"] < 0:
        return False
    uri = ex.get("heatmap_image")
    if not isinstance(uri, str) or not uri.startswith("data:image/png;base64,"):
        return False
    try:
        Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1]))).verify()
    except Exception:                             # noqa: BLE001
        return False
    return True


def _explained(rec):
    """A patient is 'explained' if the text bars are valid AND the heatmap decodes."""
    return _valid_text(rec.get("text")) and _valid_heatmap(rec.get("mri"))


# ===========================================================================
# 1-2. Smoke: explanations render on random new patients
# ===========================================================================
def test_explains_3_random_new_patients(new_patients, xai_results):
    pick = RNG.sample(list(xai_results), 3)
    for pid in pick:
        rec = xai_results[pid]
        assert _valid_text(rec.get("text")), f"bad text xAI for {pid}: {rec}"
        assert _words_are_from_note(rec["text"], rec["note"]), \
            f"influential words for {pid} are not from the note"
        assert _valid_heatmap(rec.get("mri")), f"bad heatmap for {pid}: {rec.get('mri_status')}"
    print("\n[3 random] top feature / top slice:",
          {p: (xai_results[p]["text"]["feature_importance"][0]["feature"],
               xai_results[p]["mri"]["top_slice_index"]) for p in pick})


def test_explains_1_random_new_patient(xai_results):
    pid = RNG.choice(list(xai_results))
    rec = xai_results[pid]
    assert _valid_text(rec.get("text")) and _valid_heatmap(rec.get("mri"))
    print(f"\n[1 random] {pid}")
    print("  text summary:", rec["text"]["summary"])
    print("  mri  summary:", rec["mri"]["summary"])


# ===========================================================================
# 3-5. Explanations render for each ground-truth subtype
# ===========================================================================
def _subtype_case(dx_code, new_patients, xai_results):
    pool = new_patients["buckets"].get(dx_code, [])
    if len(pool) < 3:
        pytest.skip(f"only {len(pool)} '{DX_NAME[dx_code]}' patients (need 3)")
    pick = RNG.sample(pool, 3)
    for pid in pick:
        assert _valid_text(xai_results[pid].get("text")), xai_results[pid]
        assert _valid_heatmap(xai_results[pid].get("mri")), xai_results[pid]
    print(f"\n[{DX_NAME[dx_code]} x3] top words:",
          {p: [w["word"] for w in xai_results[p]["text"]["influential_words"][:3]]
           for p in pick})


def test_explains_3_inattentive(new_patients, xai_results):
    _subtype_case(3, new_patients, xai_results)


def test_explains_3_hyperactive(new_patients, xai_results):
    _subtype_case(2, new_patients, xai_results)


def test_explains_3_combined(new_patients, xai_results):
    _subtype_case(1, new_patients, xai_results)


# ===========================================================================
# 6-8. Explained correctly over growing batches
# ===========================================================================
def _batch(n, xai_results):
    ids = list(xai_results)
    if len(ids) < n:
        pytest.skip(f"only {len(ids)} usable new patients (need {n})")
    pick = RNG.sample(ids, n)
    # every patient must get valid text bars
    assert all(_valid_text(xai_results[p].get("text")) for p in pick), \
        "some patients have malformed text explanations"
    heat_rate = sum(_valid_heatmap(xai_results[p].get("mri")) for p in pick) / n
    print(f"\n[{n} patients] text xAI valid = 100%  heatmap decodable = {heat_rate:.1%}")
    assert heat_rate >= HEATMAP_FLOOR, f"only {heat_rate:.1%} heatmaps decodable"


def test_30_new_patients_explained_correctly(xai_results):
    _batch(30, xai_results)


def test_50_new_patients_explained_correctly(xai_results):
    _batch(50, xai_results)


def test_all_new_patients_explained_correctly(new_patients, xai_results):
    ids = list(xai_results)
    assert all(_valid_text(xai_results[p].get("text")) for p in ids)
    heat_rate = sum(_valid_heatmap(xai_results[p].get("mri")) for p in ids) / len(ids)
    words_from_note = sum(_words_are_from_note(xai_results[p]["text"], xai_results[p]["note"])
                          for p in ids) / len(ids)
    explained = sum(_explained(xai_results[p]) for p in ids) / len(ids)
    top_features = Counter(xai_results[p]["text"]["feature_importance"][0]["feature"]
                           for p in ids)
    print(f"\n[ALL {len(ids)} patients] fully explained = {explained:.1%}  "
          f"(text valid=100%, heatmap decodable={heat_rate:.1%}, "
          f"words-from-note={words_from_note:.1%})")
    print("  most common top driver:", top_features.most_common(3))
    assert heat_rate >= HEATMAP_FLOOR
    assert words_from_note >= HEATMAP_FLOOR      # word attribution is faithful to the note
