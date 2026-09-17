#!/usr/bin/env python3
"""Work-dependence regression test for linear_op op timing (debug-loop handoff, TESTS section).

Legacy assertions (the RED signature of host-bound CUDA-event timing; they fail on jobs 21313/21334/21377):
  T1  attn_post_proj median(64 tok) / median(8 tok)   >= 1.30 at every TP
  T2  attn_post_proj median(4096 tok) / median(64 tok) >= 2.00 at TP 1, 2, 4
  T3  1-token attn_post_proj median max/min across TP  >= 1.5
T1 and T3 were found to be mis-specified for a correct measurement (kernel traces: the 8->64-token GEMM grows only 1.10-1.17x
at TP1/TP2 because it is weight-read bound; the 1-token kernel is fixed-cost bound, spread 1.8x). The user accepted (2026-09-17)
the revised acceptance test for the GPU-bound column:
  T1' legacy_host_bound_ratio.attn_post_proj >= 2.0 at 8 and 64 tokens at every TP (observed 2.77-5.20)
      and time_stats.attn_post_proj.median <= 1.05 * time_stats_hostbound.attn_post_proj.median on every row (observed max 1.020)
  T2  unchanged, on time_stats (observed 4.77/3.11/2.11; TP4 margin 3-7 % is a known flake risk: report, do not re-threshold)
Monotonicity is deliberately NOT tested. All modes need rows at tokens {1, 8, 64, 4096} x TP {1, 2, 4, 8}; a missing cell is
reported as "missing row" and fails. The "legacy T1/T2/T3 must FAIL on time_stats_hostbound" half of --expect green is a weak
check (legacy T1 also fails on a correct measurement); the decisive part of GREEN is T1'/T2 on time_stats.

Usage:
  test_measurement_validity.py --expect red   <linear_op.csv>   single-column (legacy) file: exit 0 iff T1/T2/T3 FAIL on time_stats
  test_measurement_validity.py --expect green <linear_op.csv>   two-column file: exit 0 iff T1'/T2 pass on time_stats AND
                                                                 T1/T2/T3 fail on time_stats_hostbound
  test_measurement_validity.py <linear_op.csv>                   legacy behaviour: run T1/T2/T3 on time_stats, exit 1 on failure
"""
import argparse, sys
import pandas as pd

OP = "attn_post_proj"
TPS = (1, 2, 4, 8)
MISSING = []  # (tp, tok, col) cells absent from the file; any entry fails the run in every mode


def _med(df, col, tp, tok):
    r = df[(df.num_tensor_parallel_workers == tp) & (df.num_tokens == tok)][col] if col in df.columns else []
    if not len(r):
        print(f"  missing row: TP{tp} tokens={tok} column={col}")
        MISSING.append((tp, tok, col))
        return float("nan")
    return float(r.iloc[0])


def legacy_assertions(df, col):
    """T1/T2/T3 on `col`; returns list of (name, value, threshold, passed)."""
    out = []
    for tp in TPS:
        v = _med(df, col, tp, 64) / _med(df, col, tp, 8); out.append((f"T1 TP{tp} 64tok/8tok", v, 1.30, v >= 1.30))
    for tp in (1, 2, 4):
        v = _med(df, col, tp, 4096) / _med(df, col, tp, 64); out.append((f"T2 TP{tp} 4096tok/64tok", v, 2.00, v >= 2.00))
    one = [_med(df, col, tp, 1) for tp in TPS]
    v = max(one) / min(one); out.append(("T3 1tok max/min across TP", v, 1.5, v >= 1.5))
    return out


def revised_assertions(df):
    """T1'/T2 on the GPU-bound column of a two-column file."""
    gpu, legacy, ratio = f"time_stats.{OP}.median", f"time_stats_hostbound.{OP}.median", f"legacy_host_bound_ratio.{OP}"
    out = []
    for tp in TPS:
        for tok in (8, 64):
            v = _med(df, ratio, tp, tok); out.append((f"T1' TP{tp} legacy/GPU-bound @{tok}tok", v, 2.0, v >= 2.0))
    worst = (df[gpu] / df[legacy]).max()
    out.append(("T1' max GPU-bound/legacy over all rows", float(worst), 1.05, worst <= 1.05))
    for tp in (1, 2, 4):
        v = _med(df, gpu, tp, 4096) / _med(df, gpu, tp, 64); out.append((f"T2 TP{tp} 4096tok/64tok (GPU-bound)", v, 2.00, v >= 2.00))
    return out


def report(title, rows):
    print(f"== {title}")
    for name, v, thr, ok in rows:
        print(f"  {name} = {v:.3f}  {'PASS' if ok else 'FAIL'} (threshold {thr})")
    return all(ok for *_, ok in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--expect", choices=["red", "green"], default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, low_memory=False, float_precision="round_trip")
    two_column = f"time_stats_hostbound.{OP}.median" in df.columns
    if a.expect is None:
        ok = report(f"legacy T1/T2/T3 on time_stats ({a.csv})", legacy_assertions(df, f"time_stats.{OP}.median"))
        verdict, code = ("GREEN", 0) if ok else ("RED", 1)
    elif a.expect == "red":
        if two_column:
            print("RESULT: two-column file given with --expect red; the legacy column is time_stats_hostbound, use --expect green"); return 1
        all_pass = report("legacy T1/T2/T3 on time_stats (expected to FAIL: host-bound signature)", legacy_assertions(df, f"time_stats.{OP}.median"))
        verdict, code = ("RED as expected", 0) if not all_pass else ("UNEXPECTED GREEN (legacy assertions passed)", 1)
    else:  # green
        if not two_column:
            print("RESULT: --expect green needs the two-column schema (time_stats_hostbound.*); file is single-column"); return 1
        fixed_ok = report("T1'/T2 on the GPU-bound column (expected PASS)", revised_assertions(df))
        legacy_all_pass = report("legacy T1/T2/T3 on time_stats_hostbound (expected FAIL: the RED signature must persist in the legacy column)",
                                 legacy_assertions(df, f"time_stats_hostbound.{OP}.median"))
        verdict, code = ("GREEN as expected", 0) if (fixed_ok and not legacy_all_pass) else ("FAIL", 1)
    if MISSING:
        print(f"RESULT: FAIL - {len(MISSING)} required cell(s) missing (tokens {{1,8,64,4096}} x TP {{1,2,4,8}} needed): {MISSING[:4]}"); return 1
    print("RESULT:", verdict)
    return code


if __name__ == "__main__":
    sys.exit(main())
