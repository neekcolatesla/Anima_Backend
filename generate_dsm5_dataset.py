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
def _severity(raw_sum):
    if raw_sum <= 9:
        return "minimal"
    if raw_sum <= 18:
        return "mild"
    if raw_sum <= 27:
        return "moderate"
    return "marked"


def _pronouns(gender):
    # ADHD-200 Gender: 1 = Male, 0 = Female.
    if gender == 1:
        return ("he", "him", "his")
    if gender == 0:
        return ("she", "her", "her")
    return ("they", "them", "their")


def generate_notes(dx, is_child, gender, ia_raw, hi_raw, rng):
    subj = "the student" if is_child else "the patient"
    he, him, his = _pronouns(gender)

    if is_child:
        openers = ["The parent and classroom teacher report that",
                   "According to guardian and teacher observations,",
                   "Home and school reports indicate that",
                   "Caregiver and teacher feedback notes that"]
        impact_adhd = ["which is disrupting schoolwork and daily routines.",
                       "affecting classroom participation and homework completion.",
                       "with a clear impact on learning and peer interactions."]
        impact_ctrl = ["with no notable impact on schoolwork or behaviour.",
                       "and functioning at school appears age-appropriate."]
    else:
        openers = ["The patient self-reports that",
                   "During the clinical interview, the patient described that",
                   "Self-report and clinician observation indicate that",
                   "The patient and a close informant report that"]
        impact_adhd = ["which is interfering with work performance and daily organisation.",
                       "affecting occupational functioning and time management.",
                       "with a clear impact on work and personal responsibilities."]
        impact_ctrl = ["with no notable impact on work or daily functioning.",
                       "and overall functioning appears within normal limits."]

    opener = rng.choice(openers)

    if dx == 0:  # control
        body = (f"{subj} shows attention and activity levels within the expected "
                f"range for {his} age, with only occasional, situational lapses")
        impact = rng.choice(impact_ctrl)
        return f"{opener} {body} {impact}"

    ia_sev, hi_sev = _severity(ia_raw), _severity(hi_raw)
    inatt_txt = (f"{ia_sev} inattentive symptoms - {he} is easily distracted, "
                 f"loses focus on tasks, and is often forgetful")
    hyper_txt = (f"{hi_sev} hyperactive-impulsive symptoms - {he} is restless, "
                 f"fidgety, and struggles to wait {his} turn")

    if dx == 3:      # inattentive
        symptoms = inatt_txt
    elif dx == 2:    # hyperactive/impulsive
        symptoms = hyper_txt
    else:            # combined
        symptoms = f"{inatt_txt}, together with {hi_sev} hyperactive-impulsive behaviour"

    impact = rng.choice(impact_adhd)
    label = DX_LABELS.get(dx, "ADHD")
    return (f"{opener} {subj} presents with {symptoms}, {impact} "
            f"Presentation is consistent with {label}.")


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

        # 5. Diagnosis- and age-appropriate clinician/guardian notes.
        notes = generate_notes(dx if dx is not None else 0, is_child, gender,
                               sum(ia_answers), sum(hi_answers), rng)

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