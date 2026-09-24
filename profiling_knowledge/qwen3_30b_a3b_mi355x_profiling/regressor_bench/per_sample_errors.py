"""Per-sample error file: one row per (model, op, TP, test shape, collection, timed sample).

Reads the shape-level test predictions of baseline v2 (interp, RF, XGBoost exact) and expands every test shape into
its individual timed samples from two collections: 21539 (settle 8, the training collection) and 21519 (settle 3).
Timed samples are the ``time_stats.<op>.samples`` JSON with the first ``warmup_count`` entries dropped;
``sample_index`` is the run position starting at 1 (1..25; 1..50 for emb, which is timed twice per forward).
The prediction is per shape and repeats across that shape's samples. No aggregation, no metrics.

Test tier only. The holdout is never read.

Output: results/baseline_v2/per_sample_errors.csv and per_sample_row_counts.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from regressor_bench.dataset import DEFAULT_CSV, TOKENS_COL, TP_COL, md5_of  # noqa: E402
from regressor_bench.splits import load_split  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "baseline_v2"
COLLECTIONS = {
    "21539": (Path(DEFAULT_CSV), "3635e4d305ae5cf84884e275d3ff37ef"),
    "21519": (Path("data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-22_job21519_dense_workbacklog_settle3/linear_op.csv"),
              "7016bd6260aacf40824122d09f627a44"),
}


def main() -> int:
    preds = pd.read_csv(OUT / "test_predictions.csv")
    preds = preds[preds.tier == "test"][["model", "op", "tp", "num_tokens", "pred"]]
    split = load_split(HERE / "splits", "random", include_holdout=False)
    test_tokens = set(map(int, split["test"]))
    assert set(preds.num_tokens.unique()) == test_tokens, "prediction file does not cover exactly the test tier"
    ops = sorted(preds.op.unique())

    frames = []
    for coll, (path, md5) in COLLECTIONS.items():
        assert md5_of(path) == md5, f"{path}: unexpected md5"
        cols = [TOKENS_COL, TP_COL] + [f"time_stats.{op}.{k}" for op in ops for k in ("samples", "warmup_count", "count")]
        raw = pd.read_csv(path, low_memory=False, usecols=cols)
        raw = raw[raw[TOKENS_COL].isin(test_tokens)]
        for op in ops:
            sub = raw.loc[raw[f"time_stats.{op}.samples"].notna(),
                          [TOKENS_COL, TP_COL, f"time_stats.{op}.samples", f"time_stats.{op}.warmup_count", f"time_stats.{op}.count"]]
            sub.columns = ["num_tokens", "tp", "samples", "warmup_count", "count"]
            for r in sub.itertuples(index=False):
                samples = json.loads(r.samples)
                warm, count = int(r.warmup_count), int(r.count)
                timed = samples[warm:]
                assert len(timed) == count, (coll, op, r.num_tokens, r.tp, len(samples), warm, count)
                frames.append(pd.DataFrame({
                    "op": op, "tp": int(r.tp), "num_tokens": int(r.num_tokens), "collection": coll,
                    "sample_index": np.arange(1, count + 1), "measured_ms": np.asarray(timed, dtype=float),
                }))
    samples = pd.concat(frames, ignore_index=True)
    out = preds.merge(samples, on=["op", "tp", "num_tokens"], how="inner").rename(columns={"pred": "predicted_ms"})
    out = out[["model", "op", "tp", "num_tokens", "collection", "sample_index", "measured_ms", "predicted_ms"]]
    out = out.sort_values(["model", "op", "tp", "num_tokens", "collection", "sample_index"]).reset_index(drop=True)
    out.to_csv(OUT / "per_sample_errors.csv", index=False)

    counts = out.groupby(["model", "op", "tp"]).agg(rows=("measured_ms", "size"), shapes=("num_tokens", "nunique"),
                                                    collections=("collection", "nunique"),
                                                    samples_per_shape_per_collection=("sample_index", "max")).reset_index()
    counts["expected_rows"] = counts.shapes * counts.collections * counts.samples_per_shape_per_collection
    counts["complete"] = counts.rows == counts.expected_rows
    counts.to_csv(OUT / "per_sample_row_counts.csv", index=False)
    pd.set_option("display.width", 200)
    print(counts.to_string(index=False))
    print(f"\ntotal rows {len(out):,}  test shapes {out.num_tokens.nunique()}  all complete: {bool(counts.complete.all())}")
    print(f"-> {OUT / 'per_sample_errors.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
