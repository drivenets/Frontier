"""Two-column (legacy + GPU-bound) op timing for linear_op profiling.

Debug-loop handoff: amd-playground/.claude/debug-reports/debug_linear_op_host_bound_timing_2026-09-16.md (Stage 5).
Root cause: CUDA-event scopes in an un-synchronised host-bound loop measure host launch span, not kernel time
(cuda_timer.py:54-56/93-97, linear_op_wrapper.py timed loop). Fix: time every shape twice and emit both columns plus a
legacy/GPU-bound ratio flag. CPU tests cover the pure helpers and the CSV pipeline; GPU tests run the wrapper itself.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # the wrapper imports torch at module level

from frontier.profiling.linear_op.linear_op_wrapper import (  # noqa: E402
    BACKLOG_COVERAGE_FACTOR,
    LEGACY_HOST_BOUND_RATIO_THRESHOLD,
    compute_legacy_host_bound,
)
from frontier.profiling.linear_op.main import expand_dict_columns  # noqa: E402
from frontier.profiling.utils.replicated_ops import split_replicated_result  # noqa: E402
REPO = Path(__file__).resolve().parents[2]
VALIDITY = REPO / "profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/test_measurement_validity.py"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU; run inside the ROCm container on a cluster node")


def _stats(**medians):
    return {op: {"median": m, "count": 50, "warmup_count": 3} for op, m in medians.items()}


# ---------------------------------------------------------------- CPU: pure helpers

def test_legacy_host_bound_ratio_and_flag():
    gpu_bound = _stats(attn_post_proj=0.0100, attn_rope=0.0500, forward_gpu_span=0.3)
    legacy = _stats(attn_post_proj=0.0360, attn_rope=0.0505)
    ratio, flag = compute_legacy_host_bound(legacy, gpu_bound)
    assert ratio == {"attn_post_proj": pytest.approx(3.6), "attn_rope": pytest.approx(1.01)}
    assert flag == {"attn_post_proj": True, "attn_rope": False}
    assert "forward_gpu_span" not in ratio  # whole-forward span has no legacy counterpart
    assert LEGACY_HOST_BOUND_RATIO_THRESHOLD == 1.15  # basis: legacy/GPU-bound 0.98-1.01 where idle gap <= 6 us, >= 1.60 where >= 40 us


def test_legacy_host_bound_handles_zero_and_missing():
    ratio, flag = compute_legacy_host_bound(_stats(a=0.0, b=0.02), _stats(a=0.0, c=0.01))
    assert ratio == {"a": pytest.approx(1.0)} and flag == {"a": False}  # 0/0 -> 1.0 (no evidence), unmatched ops dropped


def test_split_replicated_result_splits_every_per_op_dict():
    result = {
        "num_tensor_parallel_workers": 2,
        "time_stats": _stats(emb=0.05, attn_post_proj=0.03),
        "time_stats_hostbound": _stats(emb=0.09, attn_post_proj=0.06),
        "legacy_host_bound_ratio": {"emb": 1.8, "attn_post_proj": 2.0},
        "legacy_host_bound": {"emb": True, "attn_post_proj": True},
        "host_wall_per_forward_ms": 0.6,
    }
    sharded, replicated = split_replicated_result(result, {"emb"})
    for key in ("time_stats", "time_stats_hostbound", "legacy_host_bound_ratio", "legacy_host_bound"):
        assert set(sharded[key]) == {"attn_post_proj"}, key
        assert set(replicated[key]) == {"emb"}, key
    assert replicated["num_tensor_parallel_workers"] == 1 and sharded["host_wall_per_forward_ms"] == 0.6


def test_expand_dict_columns_flattens_all_dict_fields():
    df = pd.DataFrame([{
        "num_tokens": 64,
        "time_stats": _stats(attn_post_proj=0.010),
        "time_stats_hostbound": _stats(attn_post_proj=0.036),
        "legacy_host_bound_ratio": {"attn_post_proj": 3.6},
        "legacy_host_bound": {"attn_post_proj": True},
        "gpu_backlog_ms_actual": 78.1,
    }])
    out = expand_dict_columns(df)
    for col in ("time_stats.attn_post_proj.median", "time_stats_hostbound.attn_post_proj.median",
                "legacy_host_bound_ratio.attn_post_proj", "legacy_host_bound.attn_post_proj", "gpu_backlog_ms_actual", "num_tokens"):
        assert col in out.columns, col
    assert not any(isinstance(v, dict) for v in out.iloc[0].tolist())
    assert out.loc[0, "legacy_host_bound_ratio.attn_post_proj"] == pytest.approx(3.6)


# ---------------------------------------------------------------- CPU: validity script --expect modes

def _write_csv(path, two_column):
    rows = []
    kernel = {1: 0.0059, 8: 0.0107, 64: 0.0120, 4096: 0.0372}  # GPU-bound medians observed in job 21378 (TP2)
    legacy = {1: 0.0215, 8: 0.0347, 64: 0.0366, 4096: 0.0615}  # legacy medians observed in job 21377 (TP2)
    for tp in (1, 2, 4, 8):
        for tok, k in kernel.items():
            row = {"num_tokens": tok, "num_tensor_parallel_workers": tp}
            if two_column:
                row["time_stats.attn_post_proj.median"] = k
                row["time_stats_hostbound.attn_post_proj.median"] = legacy[tok]
                row["legacy_host_bound_ratio.attn_post_proj"] = legacy[tok] / k
            else:
                row["time_stats.attn_post_proj.median"] = legacy[tok]
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize("two_column,expect,exit_code", [(False, "red", 0), (True, "green", 0), (False, "green", 1), (True, "red", 1)])
def test_validity_script_expect_modes(tmp_path, two_column, expect, exit_code):
    csv = tmp_path / "linear_op.csv"
    _write_csv(csv, two_column)
    proc = subprocess.run([sys.executable, str(VALIDITY), "--expect", expect, str(csv)], capture_output=True, text=True)
    assert proc.returncode == exit_code, proc.stdout + proc.stderr


# ---------------------------------------------------------------- GPU: the wrapper itself

@pytest.fixture(scope="module")
def gpu_row(tmp_path_factory):
    from frontier.profiling.common.model_config import ModelConfig
    from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
    cfg = ModelConfig.from_model_name("Qwen3-30B-A3B-tiny")
    wrapper = LinearOpWrapper(cfg, 1, "cuda_event", rank=0, output_dir=str(tmp_path_factory.mktemp("prof")))
    return wrapper.profile(64)


@gpu
def test_two_column_schema(gpu_row):
    row = gpu_row
    for key in ("time_stats", "time_stats_hostbound", "host_wall_per_forward_ms", "host_wall_per_forward_ms_backlog",
                "gpu_backlog_ms", "gpu_backlog_ms_actual", "legacy_host_bound_ratio", "legacy_host_bound"):
        assert key in row, key
    assert "forward_gpu_span" in row["time_stats"] and "forward_gpu_span" not in row["time_stats_hostbound"]
    for op in ("attn_pre_proj", "attn_rope", "attn_post_proj"):
        for col in (row["time_stats"], row["time_stats_hostbound"]):
            assert col[op]["count"] == 50 and col[op]["warmup_count"] == 3, op  # same 50 timed runs, no filtering
        assert op in row["legacy_host_bound_ratio"] and op in row["legacy_host_bound"]


@gpu
def test_backlog_covers_host_time(gpu_row):
    row = gpu_row
    assert BACKLOG_COVERAGE_FACTOR == 3.0
    assert row["gpu_backlog_ms_actual"] >= BACKLOG_COVERAGE_FACTOR * 50 * row["host_wall_per_forward_ms"]  # all terms in ms


@gpu
def test_forward_span_closure_is_recorded(gpu_row, record_property):
    """F6 whole-forward closure (debug report residual uncertainty (1)) is a MEASUREMENT, not a gate.

    On the validation grid it holds within +-3 % at >= 3072 tokens but reads 0.77-0.79 at 8 tokens (unexplained); on this
    single tiny-model shape it has both failed (job 21386, xfail) and held (job 21390, XPASS), so a strict xfail marker was
    unstable. The test therefore prints the ratio for the record and only rejects values outside a broad envelope that
    would indicate the span column or the spin accounting is broken.
    """
    row = gpu_row
    span_total = row["time_stats"]["forward_gpu_span"]["median"] * 50
    wall_minus_sleep = row["host_wall_per_forward_ms_backlog"] * 50 - row["gpu_backlog_ms_actual"]
    closure = span_total / wall_minus_sleep
    record_property("f6_closure", round(closure, 4))  # lands in junit XML regardless of capture flags
    print(f"F6 closure forward_gpu_span*50 / (backlog wall*50 - spin) = {closure:.3f} (within +-3 %: {0.97 <= closure <= 1.03})")
    assert 0.70 <= closure <= 1.03, closure
