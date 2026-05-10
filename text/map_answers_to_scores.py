#!/usr/bin/env python3
"""
Map categorical LLM answers (yes/no/to some extent/not mentioned) to numeric scores.

  yes            → 1.0
  no             → 0.0
  to some extent → 0.5
  not mentioned  → column mean (mean imputation)

Input:  CSV with patient_id and q1–q11 columns (output of parse_llm_answers.py)
Output: CSV with the same columns, values replaced by floats
"""
import csv
from pathlib import Path

# ── Configure these paths for your environment ─────────────────────────────────
IN_CSV  = Path("CONFIGURE_ME/features/llm_question_answers.csv")
OUT_CSV = Path("CONFIGURE_ME/features/llm_question_scores.csv")
# ──────────────────────────────────────────────────────────────────────────────


QCOLS = [f"q{i}" for i in range(1, 12)]

def to_score(val: str):
    if val is None: return None
    t = val.strip().lower()
    if t == "yes":               return 1.0
    if t == "no":                return 0.0
    if t == "to some extent":    return 0.5
    if t == "" or t == "not mentioned":
        return None
    return None

def main():
    # Read input
    with IN_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)

    if "patient_id" not in reader.fieldnames:
        raise ValueError("Input CSV must contain 'patient_id' column.")
    for q in QCOLS:
        if q not in reader.fieldnames:
            raise ValueError(f"Input CSV missing required column: {q}")

    # First pass: map to numeric and collect per-question stats for means
    mapped_values = []
    sums = {q: 0.0 for q in QCOLS}
    counts = {q: 0 for q in QCOLS}

    for r in rows_in:
        rec = {"patient_id": r["patient_id"]}
        for q in QCOLS:
            sc = to_score(r.get(q, ""))
            rec[q] = sc
            if sc is not None:
                sums[q] += sc
                counts[q] += 1
        mapped_values.append(rec)

    means = {}
    for q in QCOLS:
        if counts[q] > 0:
            means[q] = sums[q] / counts[q]
        else:
            means[q] = 0.5

    # Second pass: fill missing with column mean
    out_rows = []
    for rec in mapped_values:
        out_row = {"patient_id": rec["patient_id"]}
        for q in QCOLS:
            val = rec[q]
            if val is None:
                val = means[q]
            out_row[q] = val
        out_rows.append(out_row)

    # Write output
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id"] + QCOLS)
        writer.writeheader()
        for r in out_rows:
            wr = dict(r)
            for q in QCOLS:
                wr[q] = f"{float(wr[q]):.10g}"
            writer.writerow(wr)

    # Summary
    print(f"Wrote scores to: {OUT_CSV}")
    print("Per-question means used for imputation (from available numeric answers):")
    for q in QCOLS:
        print(f"  {q}: mean={means[q]:.6f}  (n={counts[q]})")

if __name__ == "__main__":
    main()
