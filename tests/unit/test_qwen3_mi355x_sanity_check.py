"""Tests for profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py on generator-built datasets."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from frontier.profiling.utils import get_attention_input_combinations, get_num_tokens_to_profile, get_true_mixed_attention_input_combinations

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py"
OPS = ("attn_kv_cache_save", "attn_prefill", "attn_decode")
BATCH = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512]
DK = [128, 512, 1024, 2048, 4096, 8192, 16384]
SAMPLES = json.dumps([2.0] * 3 + [1.0] * 50)


def _row(block: int, tp: int, batch: int, chunk: int, kv: int, is_prefill: bool, mode: str = "even") -> dict:
    row = {
        "n_embd": 2048, "n_q_head": 32, "n_kv_head": 4, "head_dim": 128, "block_size": block,
        "num_tensor_parallel_workers": tp, "max_model_len": 16384, "batch_size": batch, "prefill_chunk_size": chunk,
        "kv_cache_size": kv, "is_prefill": is_prefill, "attention_backend": "AITER", "mode": mode,
        "is_mixed_batch": False, "is_true_mixed_batch": mode == "true_mixed", "profiling_precision": "BF16",
        "measurement_type": "CUDA_EVENT", "warmup_steps": 3, "active_steps": 50,
    }
    for op in OPS:
        row[f"time_stats.{op}.median"] = 1.0 + kv / 1000 + batch / 100 + chunk / 1000 + math.sin(kv * 7 + batch * 13 + chunk) * 1e-3
        row[f"time_stats.{op}.count"] = 50
        row[f"time_stats.{op}.samples"] = SAMPLES
    return row


def _write_dataset(root: Path) -> None:
    combos = get_attention_input_combinations(16384, 1, 512, False, False, BATCH, DK, True, -1)
    tm_combos = get_true_mixed_attention_input_combinations(max_seq_len=16384, prefill_batch_sizes=[1, 2], prefill_chunk_sizes=[1024, 4096, 8192],
                                                             decode_batch_sizes=BATCH, decode_kv_cache_sizes=DK, prefill_kv_cache_size=0)
    for block in (1, 16):
        std = pd.DataFrame([_row(block, tp, a.batch_size, a.prefill_chunk_size, a.kv_cache_size, a.is_prefill)
                            for tp in (1, 2, 4, 8) for a in combos])
        tm = pd.DataFrame([dict(_row(block, tp, 1 + a.decode_batch_size, 0, 0, True, "true_mixed"), num_prefill_seqs=len(a.prefill_seq_lens),
                                prefill_seq_lens=json.dumps(a.prefill_seq_lens), decode_batch_size=a.decode_batch_size,
                                decode_kv_cache_sizes=json.dumps(a.decode_kv_cache_sizes), decode_avg_kv_cache_size=a.decode_kv_cache_sizes[0],
                                total_batch_size=len(a.prefill_seq_lens) + a.decode_batch_size)
                           for tp in (1, 2, 4, 8) for a in tm_combos])
        std.drop(columns=["is_true_mixed_batch"]).to_csv(root / f"attention_aiter_block{block}.csv", index=False)
        tm.to_csv(root / f"attention_true_mixed_aiter_block{block}.csv", index=False)
        pd.concat([std, tm]).to_csv(root / f"attention_combined_aiter_block{block}.csv", index=False)
    for name in ("attention", "attention_true_mixed", "attention_combined"):  # exactly the plan's Step 4 union
        pd.concat([pd.read_csv(root / f"{name}_aiter_block{b}.csv", low_memory=False, float_precision="round_trip") for b in (1, 16)]).to_csv(root / f"{name}.csv", index=False)
    tokens = [t for t in get_num_tokens_to_profile(16384) if t != 4000]
    pd.DataFrame([_linear_row(t, tp) for tp in (1, 2, 4, 8) for t in tokens]).to_csv(root / "linear_op.csv", index=False)


def _linear_row(t: int, tp: int) -> dict:
    """One linear_op row in the two-column timing schema (GPU-bound time_stats.* + legacy time_stats_hostbound.*);
    _legacy_schema() strips it down to the pre-2026-09-17 single-column layout."""
    row = {"n_head": 32, "n_kv_head": 4, "n_embd": 2048, "use_qk_norm": True, "num_tokens": t,
           "num_tensor_parallel_workers": tp, "warmup_steps": 3, "active_steps": 50,
           "time_stats.attn_pre_proj.median": t / 1000, "time_stats.attn_post_proj.median": t / 2000,
           "time_stats.emb.median": 0.1 if tp == 1 else float("nan")}
    row.update({"time_stats_hostbound.attn_pre_proj.median": t / 1000 * 1.5, "time_stats_hostbound.attn_post_proj.median": t / 2000 * 2.0,
                "time_stats_hostbound.attn_rope.median": 0.06, "time_stats.attn_rope.median": 0.05,
                **{f"time_stats_hostbound.{op}.count": 50 for op in ("attn_pre_proj", "attn_rope", "attn_post_proj")},
                "time_stats.forward_gpu_span.median": 0.5, "time_stats.forward_gpu_span.count": 50,
                "host_wall_per_forward_ms": 0.6, "host_wall_per_forward_ms_backlog": 3.0, "gpu_backlog_ms": 4.0 * 50 * 0.6,
                "host_enqueue_per_forward_ms": 0.55, "host_enqueue_per_forward_ms_backlog": 0.55,
                "sclk_mhz_legacy_start": 2000.0, "sclk_mhz_legacy_end": 2100.0, "sclk_mhz_backlog_start": 2400.0, "sclk_mhz_backlog_end": 2400.0,
                "gpu_backlog_ms_actual": 4.0 * 50 * 0.6,  # coverage 4x >= the 3x gate
                "legacy_host_bound_ratio.attn_pre_proj": 1.5, "legacy_host_bound.attn_pre_proj": True,
                "legacy_host_bound_ratio.attn_rope": 1.2, "legacy_host_bound.attn_rope": True,
                "legacy_host_bound_ratio.attn_post_proj": 2.0, "legacy_host_bound.attn_post_proj": True,
                # replicated op: recorded on TP=1 rows only, NaN elsewhere (must not be counted as a flag)
                "time_stats_hostbound.emb.median": 0.15 if tp == 1 else float("nan"),
                "legacy_host_bound_ratio.emb": 1.5 if tp == 1 else float("nan"), "legacy_host_bound.emb": True if tp == 1 else float("nan")})
    return row


@pytest.fixture(scope="module")
def base_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("qwen3_dataset")
    _write_dataset(root)
    return root


@pytest.fixture
def dataset(base_dataset: Path, tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    shutil.copytree(base_dataset, root)
    return root


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(root), *extra], capture_output=True, text=True, cwd=REPO_ROOT)


def _rewrite(root: Path, names, mutate) -> None:
    for name in names:
        df = pd.read_csv(root / f"{name}.csv", low_memory=False, float_precision="round_trip")
        mutate(df).to_csv(root / f"{name}.csv", index=False)


SHAPE_FILES = ("attention", "attention_combined", "attention_aiter_block1", "attention_aiter_block16",
               "attention_combined_aiter_block1", "attention_combined_aiter_block16")
TM_FILES = ("attention_true_mixed", "attention_combined", "attention_true_mixed_aiter_block1", "attention_true_mixed_aiter_block16",
            "attention_combined_aiter_block1", "attention_combined_aiter_block16")
BLOCK1_FILES = ("attention", "attention_combined", "attention_aiter_block1", "attention_combined_aiter_block1")
TM_BLOCK1_FILES = ("attention_true_mixed", "attention_combined", "attention_true_mixed_aiter_block1", "attention_combined_aiter_block1")


# a true-mixed shape the generator actually emits (1 prefill of 1024; the largest (kv, decode batch) pair)
TM_TARGET_KV, TM_TARGET_BS = max((a.decode_kv_cache_sizes[0], a.decode_batch_size) for a in get_true_mixed_attention_input_combinations(
    max_seq_len=16384, prefill_batch_sizes=[1], prefill_chunk_sizes=[1024], decode_batch_sizes=BATCH, decode_kv_cache_sizes=DK, prefill_kv_cache_size=0))


KEY = ["block_size", "num_tensor_parallel_workers", "batch_size", "prefill_chunk_size", "kv_cache_size", "is_prefill"]


def _skew_repeated_shapes(df: pd.DataFrame) -> pd.DataFrame:
    """Second measurement of every repeated grid shape reads 10 % slower -> median spread 10 %."""
    df = df.copy()
    df.loc[df.duplicated(KEY) & (df["mode"] == "even"), "time_stats.attn_prefill.median"] *= 1.10
    return df


def _is_tm_tp8_b16_max(df: pd.DataFrame) -> pd.Series:
    return ((df.get("num_prefill_seqs") == 1) & (df.num_tensor_parallel_workers == 8) & (df.block_size == 16)
            & (df.get("decode_batch_size") == TM_TARGET_BS) & (df.get("decode_avg_kv_cache_size") == TM_TARGET_KV) & (df.get("prefill_seq_lens") == "[1024]"))


def test_passes_on_generator_built_dataset(dataset: Path) -> None:
    result = _run(dataset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("PASS")
    assert "block16 TP8: 2772/2772 standard rows, 0 memory-filtered" in result.stdout
    assert "block16 TP8: 360/360 true-mixed rows, 0 memory-filtered" in result.stdout
    # 386 TP=1 rows carry the replicated emb flag; the 1158 NaN rows at TP>1 must not be counted
    assert "linear_op: two-column schema; legacy column host-bound (ratio > 1.15) on rows per op: {'attn_pre_proj': 1544, 'attn_rope': 1544, 'attn_post_proj': 1544, 'emb': 386}" in result.stdout


def _legacy_schema(df):
    return df.drop(columns=[c for c in df.columns if c.startswith(("time_stats_hostbound.", "legacy_host_bound", "time_stats.forward_gpu_span.", "sclk_mhz_", "host_enqueue_per_forward_ms"))
                            or c in ("host_wall_per_forward_ms", "host_wall_per_forward_ms_backlog", "gpu_backlog_ms", "gpu_backlog_ms_actual")])


@pytest.mark.parametrize(
    "drop, first_missing",
    [
        (["time_stats.forward_gpu_span.median", "time_stats.forward_gpu_span.count"], "time_stats.forward_gpu_span.count"),
        (["host_wall_per_forward_ms_backlog"], "host_wall_per_forward_ms_backlog"),
        (["gpu_backlog_ms"], "gpu_backlog_ms"),
        (["legacy_host_bound.attn_pre_proj", "legacy_host_bound_ratio.attn_pre_proj"], "legacy_host_bound.attn_pre_proj"),
        (["time_stats_hostbound.attn_pre_proj.median"], "time_stats_hostbound.attn_pre_proj.median"),
    ],
    ids=["forward-span", "backlog-host-wall", "requested-backlog", "pre-proj-flags", "pre-proj-hostbound-median"],
)
def test_partial_two_column_schema_is_rejected_before_gates(dataset: Path, drop, first_missing: str) -> None:
    # any two-column field present makes the FULL set mandatory; a partial schema must fail loudly, not let the gates skip
    _rewrite(dataset, ("linear_op",), lambda df: df.drop(columns=drop))
    result = _run(dataset)
    assert result.returncode != 0
    assert "FAIL: linear_op: two-column timing schema incomplete - missing" in result.stdout, result.stdout + result.stderr
    assert f"gates not run: ['{first_missing}'" in result.stdout, result.stdout
    assert "Traceback" not in result.stderr


def test_legacy_single_column_schema_is_rejected_unless_allowed(dataset: Path) -> None:
    # the pre-2026-09-17 linear_op.csv layout (only time_stats.*) is host-bound below ~5k tokens at TP>1 and must be opted into
    _rewrite(dataset, ("linear_op",), _legacy_schema)
    result = _run(dataset)
    assert result.returncode != 0
    assert "FAIL: linear_op: single-column (legacy) timing schema" in result.stdout, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    result = _run(dataset, "--allow-legacy-schema")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "linear_op: legacy single-column schema accepted (--allow-legacy-schema)" in result.stdout


def test_non_cuda_event_run_with_nan_backlog_scalars_is_legacy_schema(dataset: Path) -> None:
    # kineto/perf_counter/record_function runs of the new wrapper keep the four backlog/host-wall scalars as all-NaN columns
    # and have no dict-derived columns; they must be treated as legacy schema, not as an incomplete two-column run
    def nan_scalars(df):
        # exactly what LinearOpWrapper.profile() emits for kineto/perf_counter: the legacy pass ran (host_wall, host_enqueue and the two
        # legacy sclk probes are real numbers), the GPU-bound pass did not (its scalars are NaN, its dict fields expand to nothing)
        df = df.drop(columns=[c for c in df.columns if c.startswith(("time_stats_hostbound.", "legacy_host_bound", "time_stats.forward_gpu_span."))])
        return df.assign(**{c: float("nan") for c in ("host_wall_per_forward_ms_backlog", "gpu_backlog_ms", "gpu_backlog_ms_actual",
                                                      "host_enqueue_per_forward_ms_backlog", "sclk_mhz_backlog_start", "sclk_mhz_backlog_end")})
    _rewrite(dataset, ("linear_op",), nan_scalars)
    result = _run(dataset)
    assert result.returncode != 0
    assert "FAIL: linear_op: single-column (legacy) timing schema" in result.stdout, result.stdout + result.stderr
    assert "two-column timing schema incomplete" not in result.stdout
    result = _run(dataset, "--allow-legacy-schema")
    assert result.returncode == 0, result.stdout + result.stderr


def test_option_in_place_of_dataset_path_is_rejected(dataset: Path) -> None:
    # regression: `sanity_check.py --data-dir <dir>` once took "--data-dir" as the dataset path, found no CSVs and printed PASS
    result = subprocess.run([sys.executable, str(SCRIPT), "--data-dir", str(dataset)], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "usage: sanity_check.py <dataset-dir>" in result.stderr + result.stdout
    assert "PASS" not in result.stdout


def test_missing_dataset_dir_fails(tmp_path: Path) -> None:
    result = _run(tmp_path / "does_not_exist")
    assert result.returncode != 0
    assert "dataset dir does not exist" in result.stderr + result.stdout


def test_empty_dataset_dir_fails_instead_of_vacuous_pass(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode != 0, result.stdout
    assert "nothing checked" in result.stdout


def test_nan_probe_value_is_rejected(dataset: Path) -> None:
    def poison(df):
        df = df.copy(); df.loc[df.index[3], "sclk_mhz_backlog_start"] = float("nan"); return df
    _rewrite(dataset, ("linear_op",), poison)
    result = _run(dataset)
    assert result.returncode != 0
    assert "FAIL: linear_op: clock-probe/enqueue columns with non-finite values (rows per column): {'sclk_mhz_backlog_start': 1}" in result.stdout, result.stdout


def test_pre_probe_two_column_run_passes_with_a_note(dataset: Path) -> None:
    # the tracked 2026-09-17 08:46 run has the two timing columns but predates the clock probe: it must still pass, with a note
    probe_cols = ["sclk_mhz_legacy_start", "sclk_mhz_legacy_end", "sclk_mhz_backlog_start", "sclk_mhz_backlog_end",
                  "host_enqueue_per_forward_ms", "host_enqueue_per_forward_ms_backlog"]
    _rewrite(dataset, ("linear_op",), lambda df: df.drop(columns=probe_cols))
    result = _run(dataset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "two-column run WITHOUT the clock probe" in result.stdout


def test_partial_probe_group_is_rejected(dataset: Path) -> None:
    _rewrite(dataset, ("linear_op",), lambda df: df.drop(columns=["sclk_mhz_backlog_end", "host_enqueue_per_forward_ms_backlog"]))
    result = _run(dataset)
    assert result.returncode != 0
    assert "FAIL: linear_op: clock-probe columns incomplete - missing ['host_enqueue_per_forward_ms_backlog', 'sclk_mhz_backlog_end']" in result.stdout, result.stdout


def test_tokens_grid_override_accepts_a_validation_grid(dataset: Path) -> None:
    grid = [1, 8, 64, 3072, 4096, 4192, 6144, 8192]
    _rewrite(dataset, ("linear_op",), lambda df: pd.DataFrame([_linear_row(t, tp) for tp in (1, 2, 4, 8) for t in grid]))
    assert _run(dataset).returncode != 0  # the default 386-value grid is required
    result = _run(dataset, "--tokens-grid", str(grid))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "linear_op: 8 token values x 4 TP = 32 rows" in result.stdout


@pytest.mark.parametrize(
    "mutate, expected_fail",
    [
        (lambda df: df.assign(gpu_backlog_ms_actual=2.0 * 50 * 0.6), "FAIL: linear_op: GPU backlog covers < 3.0x the legacy loop on 1544 rows (min ratio 2.00)"),
        (lambda df: df.assign(**{"time_stats.attn_post_proj.median": df["time_stats_hostbound.attn_post_proj.median"] * 1.10}),
         "FAIL: linear_op: GPU-bound attn_post_proj exceeds 1.05x legacy on 1544 rows (max 1.100)"),
    ],
    ids=["backlog-coverage", "gpu-bound-above-legacy"],
)
def test_two_column_hard_gates_report_exact_line(dataset: Path, mutate, expected_fail: str) -> None:
    _rewrite(dataset, ("linear_op",), mutate)
    result = _run(dataset)
    assert result.returncode != 0
    assert expected_fail in result.stdout, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("names", "mutate", "expected_fail"),
    [
        (("attention",), lambda df: df.drop(columns=["head_dim"]), "FAIL: attention: missing columns ['head_dim']"),
        (("attention",), lambda df: df.assign(attention_backend=["TORCH_SDPA"] + ["AITER"] * (len(df) - 1)), "FAIL: attention: attention_backend"),
        (SHAPE_FILES, lambda df: df[~((df.num_tensor_parallel_workers == 8) & (df.block_size == 16) & ~df.is_prefill.astype(bool)
                                      & (df.batch_size == 512) & (df.kv_cache_size == 16384))], "FAIL: block16 TP8: 1 missing decode shapes (not explained by the KV memory budget)"),
        (BLOCK1_FILES, lambda df: pd.concat([df, df[(df.block_size == 1) & (df["mode"] == "even")].iloc[[0]].assign(batch_size=3)]),
         "FAIL: block1 TP1: 1 unplanned shapes"),
        (SHAPE_FILES, lambda df: df[~((df.num_tensor_parallel_workers == 8) & (df.block_size == 16) & df.is_prefill.astype(bool)
                                      & (df.prefill_chunk_size == 16384) & (df.kv_cache_size == 0) & (df["mode"] == "even"))],
         "FAIL: block16 TP8: missing prefill shapes"),
        (TM_FILES, lambda df: df[~_is_tm_tp8_b16_max(df)], "FAIL: block16 TP8: 1 missing true-mixed shapes (not explained by the KV memory budget)"),
        (TM_BLOCK1_FILES, lambda df: pd.concat([df, df[(df.block_size == 1) & (df["mode"] == "true_mixed")].iloc[[0]].assign(decode_batch_size=3)]),
         "FAIL: block1 TP1: 1 unplanned true-mixed shapes"),
        (("attention",), lambda df: df.assign(**{"time_stats.attn_decode.count": 49}), "FAIL: attention: attn_decode.count != 50"),
        (("attention",), lambda df: df.assign(**{"time_stats.attn_prefill.samples": json.dumps([1.0] * 52)}), "FAIL: attention: attn_prefill.samples not 53 long"),
        (("attention",), lambda df: df[df.block_size == 16], "FAIL: attention: block/TP sets [16] [1, 2, 4, 8]"),
        (("attention_true_mixed",), lambda df: df.drop(columns=["decode_batch_size"]), "FAIL: attention_true_mixed: missing columns ['decode_batch_size']"),
        (SHAPE_FILES, _skew_repeated_shapes,
         "FAIL: repeated shapes disagree"),
        (("attention_aiter_block1",), lambda df: df.iloc[1:], "FAIL: attention: canonical content != block1 + block16"),
        (("attention",), lambda df: df.assign(**{"time_stats.attn_decode.median": df["time_stats.attn_decode.median"] * 1.01}),
         "FAIL: attention: canonical content != block1 + block16"),
        (("attention_combined",), lambda df: df.assign(**{"time_stats.attn_decode.median": df["time_stats.attn_decode.median"] * 1.01}),
         "FAIL: combined != attention + true_mixed"),
        (SHAPE_FILES, lambda df: df[~((df.num_tensor_parallel_workers == 8) & (df.block_size == 16) & (df["mode"] == "even"))],
         "FAIL: block16 TP8: missing prefill shapes"),
        (TM_FILES, lambda df: df[~((df.num_tensor_parallel_workers == 8) & (df.block_size == 16) & (df["mode"] == "true_mixed"))],
         "FAIL: block16 TP8: 360 missing true-mixed shapes (not explained by the KV memory budget)"),
        (("attention",), lambda df: df.assign(max_model_len=8192), "FAIL: attention: max_model_len/precision/measurement"),
        (("attention",), lambda df: df.assign(profiling_precision="FP16"), "FAIL: attention: max_model_len/precision/measurement"),
        (("attention",), lambda df: df.assign(warmup_steps=2), "FAIL: attention: warmup/active steps"),
        (("attention",), lambda df: df.assign(**{"time_stats.attn_decode.median": df["time_stats.attn_decode.median"].where(df.is_prefill.astype(bool))}),
         "FAIL: attention: NaN medians"),
        (("attention",), lambda df: df.assign(**{"time_stats.attn_kv_cache_save.median": float("nan")}), "FAIL: attention: NaN medians"),
        (("attention",), lambda df: df.assign(n_embd=4096), "FAIL: attention: model dims wrong"),
        (SHAPE_FILES, lambda df: df[~((df.num_tensor_parallel_workers == 1) & (df.block_size == 1) & (df["mode"] == "even") & ~df.is_prefill.astype(bool))],
         "FAIL: block1 TP1: smallest decode shape missing"),
        (TM_FILES, lambda df: df[~((df.num_tensor_parallel_workers == 1) & (df.block_size == 1) & (df["mode"] == "true_mixed"))],
         "FAIL: block1 TP1: smallest true-mixed shape missing"),
    ],
    ids=["missing-head_dim", "wrong-backend", "missing-decode-shape-tp8", "unplanned-shape", "missing-prefill-shape-tp8",
         "missing-true-mixed-shape-tp8", "unplanned-true-mixed-shape", "count-not-50", "samples-not-53", "block-set", "true-mixed-column-missing",
         "repeated-shape-disagrees", "canonical-vs-cell-mismatch", "canonical-stale-same-length", "combined-stale-same-length",
         "whole-std-cell-missing", "whole-true-mixed-cell-missing", "max_model_len", "precision", "warmup-steps", "decode-median-nan",
         "kv-cache-save-median-nan", "attention-n_embd", "all-decode-missing-tp1", "all-true-mixed-missing-tp1"],
)
def test_fails_with_exact_message_and_no_traceback(dataset: Path, names, mutate, expected_fail: str) -> None:
    _rewrite(dataset, names, mutate)
    result = _run(dataset)
    assert result.returncode != 0
    assert expected_fail in result.stdout, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("mutate", "expected_fail"),
    [
        (lambda df: df[df.num_tokens != 16384], "FAIL: linear_op: 1540 rows, 385 token values (need the 386-value base grid at every TP; missing [16384])"),
        (lambda df: df[~((df.num_tokens == 16384) & (df.num_tensor_parallel_workers == 8))], "FAIL: linear_op: 1543 rows, 386 token values"),
        (lambda df: df.assign(**{"time_stats.emb.median": 0.1}), "FAIL: linear_op: emb must be recorded on TP=1 rows only"),
        (lambda df: df.assign(n_embd=4096), "FAIL: linear_op: dims/TP/qk_norm"),
        (lambda df: df.assign(**{"time_stats.add.median": 0.01}), "FAIL: linear_op: add scope present"),
        (lambda df: df.assign(**{"time_stats.attn_post_proj.median": float("nan")}), "FAIL: linear_op: NaN in attn_pre/post_proj"),
    ],
    ids=["token-missing", "token-missing-one-tp", "emb-on-tp8", "n_embd", "add-scope-present", "post-proj-nan"],
)
def test_linear_op_failures_report_exact_line(dataset: Path, mutate, expected_fail: str) -> None:
    _rewrite(dataset, ("linear_op",), mutate)
    result = _run(dataset)
    assert result.returncode != 0
    assert expected_fail in result.stdout, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_symlinked_csv_in_dataset_dir_fails(dataset: Path) -> None:
    (dataset / "attention_extra.csv").symlink_to(dataset / "attention.csv")
    result = _run(dataset)
    assert result.returncode != 0
    assert "FAIL: symlinked csv in dataset dir" in result.stdout, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
