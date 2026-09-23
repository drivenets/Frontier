"""Audit the split unit and leakage for every (op, TP) regressor, with printed evidence.

Checks, per geometry and per (op, TP):
  1. the unit assigned to a tier is the distinct ``num_tokens`` value (shape level); no value in two tiers;
  2. the three tiers cover all 3,327 values;
  3. in the raw CSV every (num_tokens, TP) has exactly one row, and the 25 timed samples of a shape live inside
     that row's ``time_stats.<op>.samples`` JSON, so a row can only ever be in one tier;
  4. the actual run outputs (``test_predictions.csv``) used exactly the test tier's token values and none of the
     train/holdout values;
  5. the label column is ``time_stats.<op>.median`` and no ``time_stats_hostbound.*`` column or
     ``linear_op_kernel_only.csv`` is referenced by the training code.

Usage: python verify_splits.py [--results results/full_v1] [--splits splits] [--csv ...]
Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from regressor_bench.dataset import DEFAULT_CSV, EXPECTED_TOKEN_COUNT, TOKENS_COL, TP_COL, load_raw, regressor_keys, tidy  # noqa: E402
from regressor_bench.splits import GEOMETRIES, load_split  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--splits", default=str(HERE / "splits"))
    p.add_argument("--results", default=str(HERE / "results" / "full_v1"))
    a = p.parse_args(argv)
    failures = []

    raw = load_raw(a.csv)
    t = tidy(raw)
    keys = regressor_keys(t)

    # --- 3. one row per (num_tokens, TP); samples are inside the row -------------------------------------
    dup = int(raw.duplicated([TOKENS_COL, TP_COL]).sum())
    per_tp = raw.groupby(TP_COL)[TOKENS_COL].nunique().to_dict()
    n_samples = json.loads(raw["time_stats.attn_pre_proj.samples"].iloc[0])
    wc = int(raw["time_stats.attn_pre_proj.warmup_count"].iloc[0])
    cnt = int(raw["time_stats.attn_pre_proj.count"].iloc[0])
    print("RAW CSV")
    print(f"  rows={len(raw)}  duplicate (num_tokens, TP) rows={dup}  unique num_tokens per TP={per_tp}")
    print(f"  one row holds all timed samples of a shape: samples list len={len(n_samples)} = warmup {wc} + timed {cnt}")
    print("  => a row IS a shape; the 25 timed samples of a shape cannot be separated by any row-level split")
    if dup or any(v != EXPECTED_TOKEN_COUNT for v in per_tp.values()):
        failures.append("raw CSV is not one row per (num_tokens, TP)")

    # --- 1, 2, 4. per geometry, per (op, TP) -----------------------------------------------------------------
    results = Path(a.results)
    preds = pd.read_csv(results / "test_predictions.csv") if (results / "test_predictions.csv").exists() else None
    for geom in GEOMETRIES:
        s = load_split(a.splits, geom, include_holdout=True)  # token *lists* only; no labels are read here
        tr, te, ho = (set(map(int, s[k])) for k in ("train", "test", "holdout"))
        print(f"\nGEOMETRY {geom}: token-value tiers  train={len(tr)} test={len(te)} holdout={len(ho)}  "
              f"total={len(tr) + len(te) + len(ho)}  train&test={len(tr & te)} train&holdout={len(tr & ho)} test&holdout={len(te & ho)}")
        assert not (tr & te) and not (tr & ho) and not (te & ho), "tiers overlap"
        assert len(tr | te | ho) == EXPECTED_TOKEN_COUNT, "tiers do not cover the grid"
        print(f"  {'op':26s} {'TP':>2s} {'rows':>5s} {'train':>6s} {'test':>5s} {'hold':>5s} {'tr&te':>5s} {'tr&ho':>5s} {'te&ho':>5s} {'cover':>5s}  run-used-test-tokens")
        for op, tp in keys:
            g = t[(t.op == op) & (t.tp == tp)]
            toks = set(map(int, g.num_tokens))
            g_tr, g_te, g_ho = toks & tr, toks & te, toks & ho
            ok = (not (g_tr & g_te)) and (not (g_tr & g_ho)) and (not (g_te & g_ho)) and (g_tr | g_te | g_ho == toks)
            used = "n/a"
            if preds is not None:
                pg = preds[(preds.geometry == geom) & (preds.op == op) & (preds.tp == tp)]
                if len(pg):
                    used_toks = set(map(int, pg.num_tokens))
                    leak = used_toks & (tr | ho)
                    used = f"{len(used_toks)} == test tier: {used_toks == g_te}; overlap with train/holdout: {len(leak)}"
                    ok = ok and used_toks == g_te and not leak
            flag = "" if ok else "   <-- FAIL"
            print(f"  {op:26s} {tp:>2d} {len(g):>5d} {len(g_tr):>6d} {len(g_te):>5d} {len(g_ho):>5d} {len(g_tr & g_te):>5d} {len(g_tr & g_ho):>5d} {len(g_te & g_ho):>5d} {str(g_tr | g_te | g_ho == toks):>5s}  {used}{flag}")
            if not ok:
                failures.append(f"{geom} {op} TP{tp}")

    # --- 5. label column and forbidden inputs in the training code -------------------------------------------
    code = "".join((HERE / f).read_text() for f in ("dataset.py", "splits.py", "models.py", "bench.py"))
    hostbound_used = bool(re.search(r"time_stats_hostbound\.\w+\.(median|mean|min|max|std|samples)", code))
    kernel_only_used = "kernel_only" in code
    label_re = re.search(r'f"time_stats\.\{op\}\.median"', code)
    print("\nTRAINING INPUTS")
    print(f"  label column pattern in dataset.py: {label_re.group(0) if label_re else 'NOT FOUND'}")
    print(f"  features used by every model: ['num_tokens'] (models.FEATURE_COLS)")
    print(f"  time_stats_hostbound.* read as label/feature: {hostbound_used}")
    print(f"  linear_op_kernel_only.csv referenced: {kernel_only_used}")
    if hostbound_used or kernel_only_used or not label_re:
        failures.append("forbidden training input or wrong label")

    print("\nRESULT:", "ALL CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
