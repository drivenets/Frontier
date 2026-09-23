"""Load the Qwen3-30B-A3B / MI355X linear_op profiling CSV into a tidy per-regressor table.

Contract (from ``14_regressor_dataset_handoff.md``):

* one CSV row = one (``num_tokens``, TP) cell; every pair appears exactly once;
* the label of op ``<op>`` is ``time_stats.<op>.median`` in milliseconds;
* ``attn_pre_proj`` / ``attn_post_proj`` / ``attn_rope`` exist at TP 1, 2, 4, 8;
  ``input_layernorm`` / ``post_attention_layernorm`` / ``emb`` exist at TP 1 only (replicated ops);
* ``num_tokens`` is the only feature; every other column is a model constant or instrument metadata.

One *regressor* is one (op, tp) pair: 3 x 4 + 3 = 15 regressors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(
    "data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/"
    "2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv"
)
SHARED_STORE_CSV = (
    "/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/"
    "2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv"
)
EXPECTED_MD5 = "3635e4d305ae5cf84884e275d3ff37ef"

SHARDED_OPS: Tuple[str, ...] = ("attn_pre_proj", "attn_post_proj", "attn_rope")
REPLICATED_OPS: Tuple[str, ...] = ("input_layernorm", "post_attention_layernorm", "emb")
ALL_OPS: Tuple[str, ...] = SHARDED_OPS + REPLICATED_OPS
TP_VALUES: Tuple[int, ...] = (1, 2, 4, 8)
EXPECTED_TOKEN_COUNT = 3327
EXPECTED_ROWS = EXPECTED_TOKEN_COUNT * len(TP_VALUES)

TOKENS_COL = "num_tokens"
TP_COL = "num_tensor_parallel_workers"

# Columns that must never be used as features (instrument metadata / legacy timing), per the handoff s4.
FORBIDDEN_FEATURE_PREFIXES = (
    "time_stats_hostbound.",
    "host_wall_per_forward_ms",
    "host_enqueue_per_forward_ms",
    "gpu_backlog_ms",
    "sclk_mhz_",
    "legacy_host_bound",
)


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    return _digest(Path(path), "md5")


def sha256_of(path: Path) -> str:
    return _digest(Path(path), "sha256")


def load_raw(path: Path = DEFAULT_CSV, verify: bool = True) -> pd.DataFrame:
    """Read the profiler CSV. With ``verify`` the md5 must match the handoff document."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Mirror the shared-store file {SHARED_STORE_CSV} to that path "
            f"(md5 {EXPECTED_MD5}); data/local_datasets/ is git-ignored."
        )
    if verify:
        got = md5_of(path)
        if got != EXPECTED_MD5:
            raise ValueError(
                f"{path}: md5 {got} != expected {EXPECTED_MD5}. This harness is written for the "
                "2026-09-23 settle8 file; pass verify=False only if you know what you are doing."
            )
    return pd.read_csv(path, low_memory=False)


def median_col(op: str) -> str:
    return f"time_stats.{op}.median"


def tidy(raw: pd.DataFrame, ops: Iterable[str] = ALL_OPS) -> pd.DataFrame:
    """Long table with one row per (op, tp, num_tokens).

    Columns: ``op, tp, num_tokens, y, y_std, y_min, y_max, n_samples``; ``y`` in ms.
    Rows whose label is NaN (replicated ops at TP>1, by design) are dropped.
    """
    if raw.duplicated([TOKENS_COL, TP_COL]).any():
        raise ValueError("duplicate (num_tokens, TP) cells; the handoff promises each pair once")
    frames: List[pd.DataFrame] = []
    for op in ops:
        col = median_col(op)
        if col not in raw.columns:
            raise KeyError(f"missing label column {col}")
        d = pd.DataFrame(
            {
                "op": op,
                "tp": raw[TP_COL].astype(int),
                "num_tokens": raw[TOKENS_COL].astype(int),
                "y": raw[col].astype(float),
                "y_std": raw[f"time_stats.{op}.std"].astype(float),
                "y_min": raw[f"time_stats.{op}.min"].astype(float),
                "y_max": raw[f"time_stats.{op}.max"].astype(float),
                "n_samples": raw[f"time_stats.{op}.count"],
            }
        )
        d = d[d["y"].notna()]
        frames.append(d)
    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["op", "tp", "num_tokens"])
        .reset_index(drop=True)
    )
    _check_coverage(out)
    return out


def _check_coverage(t: pd.DataFrame) -> None:
    n_tok = t["num_tokens"].nunique()
    if n_tok != EXPECTED_TOKEN_COUNT:
        raise ValueError(f"expected {EXPECTED_TOKEN_COUNT} token values, found {n_tok}")
    for (op, tp), g in t.groupby(["op", "tp"]):
        if len(g) != EXPECTED_TOKEN_COUNT:
            raise ValueError(f"({op}, TP{tp}) has {len(g)} rows, expected {EXPECTED_TOKEN_COUNT}")
        if op in REPLICATED_OPS and tp != 1:
            raise ValueError(f"replicated op {op} has rows at TP{tp}; expected TP1 only")
    if (t["y"] <= 0).any():
        raise ValueError("non-positive label; log-target models would break")


def regressor_keys(t: pd.DataFrame) -> List[Tuple[str, int]]:
    return [(str(op), int(tp)) for op, tp in t[["op", "tp"]].drop_duplicates().itertuples(index=False)]


def token_grid(t: pd.DataFrame) -> np.ndarray:
    return np.sort(t["num_tokens"].unique())


def reference_token_grid() -> np.ndarray:
    """The profiler's grid, rebuilt from its definition (for tests that must not read the CSV)."""
    g = np.concatenate([np.arange(1, 2049), np.arange(2056, 8193, 8), np.arange(8208, 16385, 16)])
    return g[g != 4000]
