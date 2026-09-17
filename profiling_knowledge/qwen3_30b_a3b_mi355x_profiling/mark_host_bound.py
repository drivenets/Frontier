#!/usr/bin/env python3
"""Host-bound-row detector for linear_op CSVs (08_measurement_validation_preregistration.md, "Data handling").

Per row and per op it sets  host_bound_suspect.<op> = (row std > 5 x BASELINE[op]) or (|mean(runs 0-9) - mean(runs 40-49)| / median > 0.10)
with FIXED baselines pre-registered before any run (median row std of rows where job 21313 agrees with the GPU-time run 21356
within 5 %). It never edits the input: it writes <input>_flagged.csv next to it and, unless --allow-host-bound is given, exits 1
when any row is flagged (so a pipeline can assert on it). Predicted (08_): catches 60-85 % of rows known to be host-bound; misses
rows with a steady host floor (e.g. the 1-token rows). Necessary, not sufficient.
"""
import argparse, json, sys
import numpy as np, pandas as pd

BASELINE_STD_MS = {"attn_post_proj": 0.0013, "attn_rope": 0.0015, "attn_pre_proj": 0.0075}
STD_FACTOR, DRIFT_FRAC, WARMUP = 5.0, 0.10, 3

def detect(df):
    out = df.copy(); summary = {}
    for op, base in BASELINE_STD_MS.items():
        col = f"time_stats.{op}.samples"
        if col not in df:
            continue
        stds, drifts, flags = [], [], []
        for v in df[col]:
            if isinstance(v, str):
                s = np.asarray(json.loads(v), dtype=float)[WARMUP:]
                med = np.median(s); std = float(s.std())
                drift = float((s[:10].mean() - s[-10:].mean()) / med) if med > 0 else 0.0
                flags.append(std > STD_FACTOR * base or abs(drift) > DRIFT_FRAC)
            else:
                std, drift = float("nan"), float("nan"); flags.append(False)
            stds.append(std); drifts.append(drift)
        out[f"host_bound_suspect.{op}"] = flags
        out[f"host_bound_drift.{op}"] = drifts
        n = int(np.sum([isinstance(v, str) for v in df[col]]))
        summary[op] = (int(np.sum(flags)), n)
    return out, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--allow-host-bound", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, low_memory=False, float_precision="round_trip")
    out, summary = detect(df)
    dst = a.out or a.csv.replace(".csv", "_flagged.csv")
    assert dst != a.csv
    out.to_csv(dst, index=False)
    total = sum(f for f, _ in summary.values())
    for op, (f, n) in summary.items():
        print(f"{op:15s} flagged {f}/{n} rows ({100*f/max(n,1):.1f} %)")
    print(f"wrote {dst}")
    if total and not a.allow_host_bound:
        print(f"ASSERTION FAILED: {total} host-bound-suspect (op,row) entries; pass --allow-host-bound to accept", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
