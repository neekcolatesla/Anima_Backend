#!/usr/bin/env python3
"""
Anima - synthetic DSM-5 NLP training-data generator.

Reverse-engineers realistic DSM-5 ADHD questionnaire answers and clinician notes
from the *real* diagnostic scores in the NYU Athena Phenotypic CSV, and writes a
training file (DSM5_data.csv) with: patient_ID, raw_answers_json, clinician_notes.

IMPORTANT - about the scores
----------------------------
In the ADHD-200 phenotypic data the ``Inattentive`` and ``Hyper/Impulsive``
columns are Conners'-style **T-scores** (~40-90, population mean 50), NOT raw
sums of nine 0-4 items (which can only reach 36). So we can't make a 9-item sum
literally equal a T-score of, say, 90. Instead we map each T-score into the
18-item Likert space proportionally (clinical range T=40..90 -> raw 0..36) and
distribute that target across the 9 items. Higher recorded scores therefore
yield higher answers - "roughly matches" in mapped-severity terms. The original
T-scores are preserved inside each JSON under "source_scores" for traceability.

Usage
-----
    python generate_dsm5_dataset.py \
        --input NYU_Athena_Phenotypic_All.csv \
        --output DSM5_data.csv \
        --seed 42
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# 1. The 18 standard DSM-5 ADHD symptom items (child + adult wording)
#    9 Inattentive + 9 Hyperactive/Impulsive. Paraphrased from DSM-5 Criterion A.
# =============================================================================
DSM5_QUESTIONS = {
    "inattentive": [
        {"id": "IA1",
         "child": "Fails to give close attention to details or makes careless mistakes in schoolwork.",
         "adult": "Overlooks details or makes careless mistakes at work or in other activities."},
        {"id": "IA2",
         "child": "Has difficulty keeping attention on tasks, schoolwork, or play.",
         "adult": "Has difficulty sustaining attention in tasks, meetings, or lengthy reading."},
        {"id": "IA3",
         "child": "Does not seem to listen when spoken to directly.",
         "adult": "Does not seem to listen even when spoken to directly."},
        {"id": "IA4",
         "child": "Does not follow through on instructions and fails to finish schoolwork or chores.",
         "adult": "Fails to follow through on instructions and does not finish work duties."},
        {"id": "IA5",
         "child": "Has difficulty organising tasks, belongings, and activities.",
         "adult": "Has difficulty organising tasks, time, and workspace."},
        {"id": "IA6",
         "child": "Avoids or dislikes tasks that require sustained mental effort, like homework.",
         "adult": "Avoids or dislikes tasks needing sustained mental effort, like reports or forms."},
        {"id": "IA7",
         "child": "Loses things needed for tasks (e.g. pencils, books, toys).",
         "adult": "Loses things needed for tasks (e.g. keys, phone, paperwork, wallet)."},
        {"id": "IA8",
         "child": "Is easily distracted by noises or things going on around them.",
         "adult": "Is easily distracted by external stimuli or unrelated thoughts."},
        {"id": "IA9",
         "child": "Is forgetful in daily activities and routines.",
         "adult": "Is forgetful in daily activities (appointments, chores, returning calls)."},
    ],
    "hyperactive": [
        {"id": "HI1",
         "child": "Fidgets with or taps hands or feet, or squirms in their seat.",
         "adult": "Fidgets with or taps hands or feet, or squirms when seated."},
        {"id": "HI2",
         "child": "Leaves their seat when staying seated is expected (e.g. in class).",
         "adult": "Leaves their seat in situations where staying seated is expected."},
        {"id": "HI3",
         "child": "Runs about or climbs in situations where it is inappropriate.",
         "adult": "Feels restless or has difficulty sitting still for long periods."},
        {"id": "HI4",
         "child": "Is unable to play or take part in activities quietly.",
         "adult": "Is unable to engage in leisure activities quietly."},
        {"id": "HI5",
         "child": "Is often 'on the go' or acts as if driven by a motor.",
         "adult": "Feels driven to keep moving, as if propelled by a motor."},
        {"id": "HI6",
         "child": "Talks excessively.",
         "adult": "Talks excessively."},
        {"id": "HI7",
         "child": "Blurts out answers before questions have been finished.",
         "adult": "Blurts out answers or completes others' sentences before they finish."},
        {"id": "HI8",
         "child": "Has difficulty waiting for their turn.",
         "adult": "Has difficulty waiting their turn (e.g. in queues)."},
        {"id": "HI9",
         "child": "Interrupts or intrudes on others' games or conversations.",
         "adult": "Interrupts or intrudes on others (conversations, activities)."},
    ],
}

# ADHD-200 DX coding -> subtype label
DX_LABELS = {
    0: "Typically Developing (Control)",
    1: "ADHD - Combined Type",
    2: "ADHD - Hyperactive/Impulsive Type",
    3: "ADHD - Inattentive Type",
}
# Which subscale(s) each diagnosis implicates.
DX_IMPLICATES = {
    0: set(),
    1: {"inattentive", "hyperactive"},
    2: {"hyperactive"},
    3: {"inattentive"},
}

# Clinical T-score range used for the T-score -> raw-sum mapping.
T_MIN, T_MAX = 40.0, 90.0
N_ITEMS = 9          # items per subscale
MAX_LIKERT = 4       # Likert answers are 0..4


# =============================================================================
# Parsing helpers
# =============================================================================
NULL_TOKENS = {"-999", "N/A", "", "NA", "NAN", "NONE"}


def _clean(value):
    if value is None:
        return None
    s = str(value).strip()
    return None if s.upper() in NULL_TOKENS else s


def clean_int(value):
    s = _clean(value)
    if s is None:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def clean_float(value):
    s = _clean(value)
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# =============================================================================
# 3. Score -> Likert distribution helpers
# =============================================================================
def default_tscore(dx, subscale):
    """Fallback T-score when a subscale score is missing, inferred from DX."""
    if dx == 0:
        return 45.0                      # control: subclinical
    return 68.0 if subscale in DX_IMPLICATES.get(dx, set()) else 52.0


def tscore_to_raw_target(tscore):
    """Map a Conners' T-score (~40-90) to a raw 0..36 sum over 9 items (0-4)."""
    t = min(max(float(tscore), T_MIN), T_MAX)
    frac = (t - T_MIN) / (T_MAX - T_MIN)
    return int(round(frac * N_ITEMS * MAX_LIKERT))


def generate_likert(target_sum, rng, n=N_ITEMS, max_val=MAX_LIKERT):
    """Generate n integer Likert answers (0..max_val) that sum to ~target_sum.

    Produces an exact-sum vector, then applies sum-preserving jitter so the
    answers look naturally varied rather than uniform.
    """
    total = int(min(max(target_sum, 0), n * max_val))
    base, remainder = divmod(total, n)
    answers = [base] * n
    for i in rng.permutation(n)[:remainder]:
        answers[i] += 1
    # Sum-preserving jitter: move a point from one item to another.
    for _ in range(n + 3):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b and answers[a] < max_val and answers[b] > 0:
            answers[a] += 1
            answers[b] -= 1
    return [int(x) for x in answers]


# =============================================================================
# 4. Build the raw_answers JSON object for one patient
# =============================================================================
def build_raw_answers(is_child, ia_answers, hi_answers, ia_t, hi_t):
    form = "child" if is_child else "adult"
    key = "child" if is_child else "adult"

    def block(subscale, answers):
        out = {}
        for q, a in zip(DSM5_QUESTIONS[subscale], answers):
            out[q["id"]] = {"question": q[key], "answer": int(a)}
        return out

    ia_block = block("inattentive", ia_answers)
    hi_block = block("hyperactive", hi_answers)
    return {
        "instrument": "DSM-5 ADHD symptom checklist (18-item)",
        "form": form,
        "likert_scale": {"0": "Never", "1": "Rarely", "2": "Sometimes",
                         "3": "Often", "4": "Very often"},
        "inattentive": ia_block,
        "hyperactive": hi_block,
        "inattentive_sum": int(sum(ia_answers)),
        "hyperactive_sum": int(sum(hi_answers)),
        "total_sum": int(sum(ia_answers) + sum(hi_answers)),
        # Original recorded Conners' T-scores this questionnaire was derived from.
        "source_scores": {"inattentive_tscore": ia_t, "hyperactive_tscore": hi_t},
    }


# =============================================================================
# 5. Clinician / guardian notes generation
# =============================================================================
# Behaviour phrases (present-tense, so they read as "The patient <phrase>").
# The SAME vocabulary is used for every patient - ADHD and control notes differ
# only in which behaviours happen to be mentioned, never in the words available.
INATTENTIVE_PHRASES = [
    "loses focus partway through tasks",
    "is easily distracted by things nearby",
    "makes careless slips on routine work",
    "does not always seem to listen when addressed",
    "finds it hard to organise tasks and belongings",
    "avoids activities needing sustained mental effort",
    "misplaces everyday items",
    "is forgetful with daily routines",
    "leaves activities unfinished",
]
HYPERACTIVE_PHRASES = [
    "fidgets or taps when seated",
    "finds it hard to stay seated",
    "comes across as restless",
    "talks a great deal",
    "finds it hard to wait for a turn",
    "interrupts or intrudes on others",
    "seems constantly on the go",
    "finds it hard to play or work quietly",
    "acts before thinking it through",
]
# Neutral, non-diagnostic filler - present for everyone, carries NO label signal.
FILLER = [
    "Sleep and appetite were reported as unremarkable.",
    "Rapport was established easily during the session.",
    "No acute distress was observed.",
    "Family history was reviewed and is non-contributory.",
    "The informant was cooperative and engaged.",
    "General health was described as good.",
    "Mood appeared stable throughout the assessment.",
    "The session ran to time without difficulty.",
]
OPENERS_CHILD = [
    "Observations were gathered from the caregiver, teacher, and clinician.",
    "The following notes were collated during the school-age assessment.",
    "Caregiver and classroom input was reviewed alongside clinician observation.",
]
OPENERS_ADULT = [
    "Observations were gathered from self-report and clinician interview.",
    "The following notes were collated during the assessment.",
    "Self-report and an informant account were reviewed alongside clinician observation.",
]


def _mention_prob(raw_sum, signal, base=0.15):
    """Probability that a given symptom from a subscale is written into the note.

    Blends a severity-driven 'true' probability with a flat NOISE baseline that is
    identical for every patient. At signal=0.25, only a quarter of the decision
    reflects the real symptom profile and three quarters is uninformative noise,
    so the notes carry weak, realistic signal instead of restating the diagnosis.
    """
    p_true = min(max(raw_sum / (N_ITEMS * MAX_LIKERT), 0.0), 1.0)
    return signal * p_true + (1.0 - signal) * base


def _mention(phrases, raw_sum, signal, rng):
    p = _mention_prob(raw_sum, signal)
    return [ph for ph in phrases if rng.random() < p]


def generate_notes(is_child, ia_raw, hi_raw, rng, signal=0.25):
    """Symptom-level clinical note with diluted signal and NO diagnosis stated.

    The diagnosis/subtype is never named. Behaviours are mentioned probabilistically
    (mostly noise), drawn from one shared vocabulary for ADHD and control patients,
    and wrapped in neutral filler - so ADHD and control notes overlap heavily and
    the text is not a give-away for the label. `signal` (0..1) is the mixing weight
    between the true symptom profile and the flat noise baseline.
    """
    subj = "The student" if is_child else "The patient"
    openers = OPENERS_CHILD if is_child else OPENERS_ADULT

    noted = (_mention(INATTENTIVE_PHRASES, ia_raw, signal, rng)
             + _mention(HYPERACTIVE_PHRASES, hi_raw, signal, rng))
    order = rng.permutation(len(noted)) if noted else []
    noted = [noted[i] for i in order]

    parts = [str(rng.choice(openers))]
    if noted:
        shown = noted[:3]                       # keep notes short and overlapping
        if len(shown) == 1:
            parts.append(f"{subj} {shown[0]}.")
        elif len(shown) == 2:
            parts.append(f"{subj} {shown[0]} and {shown[1]}.")
        else:
            parts.append(f"{subj} " + ", ".join(shown[:-1]) + f", and {shown[-1]}.")
    else:
        parts.append("No specific attentional or behavioural concerns were raised "
                     "at this time.")

    n_filler = int(rng.integers(1, 3))          # 1-2 neutral lines, same pool for all
    for f in rng.choice(FILLER, size=n_filler, replace=False):
        parts.append(str(f))
    return " ".join(parts)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Generate synthetic DSM-5 NLP training data.")
    ap.add_argument("--input", default="NYU_Athena_Phenotypic_All.csv",
                    help="Path to the NYU Athena Phenotypic CSV.")
    ap.add_argument("--output", default="DSM5_data.csv",
                    help="Path for the generated training CSV.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    ap.add_argument("--signal", type=float, default=0.25,
                    help="Fraction of the note's symptom content driven by the true "
                         "profile vs flat noise (0..1). 0.25 = 25%% signal / 75%% noise.")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        sys.exit(f"ERROR: input file not found: {args.input}")

    required = {"ScanDir ID", "Inattentive", "Hyper/Impulsive", "DX", "Age", "Gender"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV missing required column(s): {sorted(missing)}")

    rng = np.random.default_rng(args.seed)
    rows = []

    for _, row in df.iterrows():
        # 6. patient_ID: 7-digit zero-padded ScanDir ID.
        raw_id = _clean(row["ScanDir ID"])
        if raw_id is None:
            continue
        patient_id = raw_id.split(".")[0].zfill(7)

        dx = clean_int(row["DX"])
        age = clean_float(row["Age"])
        gender = clean_int(row["Gender"])
        is_child = (age is None) or (age < 18)   # dataset is pediatric; default child

        # Recorded subscale T-scores (fall back to DX-inferred defaults if missing).
        ia_t = clean_float(row["Inattentive"])
        hi_t = clean_float(row["Hyper/Impulsive"])
        ia_t_used = ia_t if ia_t is not None else default_tscore(dx, "inattentive")
        hi_t_used = hi_t if hi_t is not None else default_tscore(dx, "hyperactive")

        # 3. Map T-scores -> raw targets -> Likert answers.
        ia_answers = generate_likert(tscore_to_raw_target(ia_t_used), rng)
        hi_answers = generate_likert(tscore_to_raw_target(hi_t_used), rng)

        # 4. Assemble the raw_answers JSON (mirrors the SQL raw_answers column).
        raw_answers = build_raw_answers(is_child, ia_answers, hi_answers, ia_t, hi_t)

        # 5. Symptom-level notes: diluted signal, diagnosis never stated.
        notes = generate_notes(is_child, sum(ia_answers), sum(hi_answers),
                               rng, args.signal)

        rows.append({
            "patient_ID": patient_id,
            "raw_answers_json": json.dumps(raw_answers, ensure_ascii=False),
            "clinician_notes": notes,
        })

    out_df = pd.DataFrame(rows, columns=["patient_ID", "raw_answers_json", "clinician_notes"])
    out_df.to_csv(args.output, index=False)

    print(f"Wrote {len(out_df)} rows to {args.output}")
    if len(out_df):
        print("\nSample record:")
        sample = out_df.iloc[0]
        print("  patient_ID:", sample["patient_ID"])
        print("  clinician_notes:", sample["clinician_notes"])
        parsed = json.loads(sample["raw_answers_json"])
        print("  form:", parsed["form"],
              "| inattentive_sum:", parsed["inattentive_sum"],
              "| hyperactive_sum:", parsed["hyperactive_sum"])


if __name__ == "__main__":
    main()