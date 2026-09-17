"""Sanity-check a Qwen3-30B-A3B MI355X dataset dir (plan Step 4).

Usage: sanity_check.py <dir> [--max-seq-len 16384] [--cells <dir holding the *_aiter_block{1,16}.csv files, default <dir>>]
                         [--tokens-grid "<python expr>"] [--allow-legacy-schema]
The attention trio is checked when attention.csv exists and linear_op.csv when it exists (a run folder may hold only one).
linear_op.csv must carry the two-column timing schema (time_stats.* GPU-bound + time_stats_hostbound.* legacy, produced only by
--profile_method cuda_event; see 07_post_proj_rope_dip_root_cause.md / 08_) unless --allow-legacy-schema is given (pre-2026-09-17
single-column runs, and kineto/perf_counter/record_function runs, which have no second pass);
--tokens-grid replaces the default 386-value token grid (e.g. the validation grid "[1,8,64,3072,4096,4192,6144,8192]")."""
import ast, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run by path without an editable install
from frontier.profiling.utils import get_attention_input_combinations, get_num_tokens_to_profile, get_true_mixed_attention_input_combinations

if len(sys.argv) < 2 or sys.argv[1].startswith("-"):  # the dataset path is positional; an option in its place once produced a vacuous PASS
    sys.exit(f"usage: sanity_check.py <dataset-dir> [--max-seq-len N] [--cells DIR] [--tokens-grid '[..]'] [--allow-legacy-schema]; got {sys.argv[1:]}")
d = Path(sys.argv[1]); fails, notes = [], []
if not d.is_dir():
    sys.exit(f"FAIL: dataset dir does not exist: {d}")
MSL = int(sys.argv[sys.argv.index("--max-seq-len") + 1]) if "--max-seq-len" in sys.argv else 16384
cells = Path(sys.argv[sys.argv.index("--cells") + 1]) if "--cells" in sys.argv else d  # per-block cell files may sit in a run folder
TOKENS_GRID = set(ast.literal_eval(sys.argv[sys.argv.index("--tokens-grid") + 1])) if "--tokens-grid" in sys.argv else None  # a literal list, e.g. "[1,8,64]"
ALLOW_LEGACY = "--allow-legacy-schema" in sys.argv
GEMM_SCOPES = ("attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj")  # the <= 1.05 gate is for GEMM scopes only (norm scopes reach 1.069, rope 1.057)
# Same values as linear_op_wrapper.BACKLOG_COVERAGE_FACTOR / the handoff's 1.05 gate; duplicated so this checker never imports torch.
BACKLOG_COVERAGE_FACTOR, GPU_BOUND_MAX_OVER_LEGACY = 3.0, 1.05
ATTN_OPS = ("attn_pre_proj", "attn_rope", "attn_post_proj")
# The complete two-column timing schema written by LinearOpWrapper.profile() under --profile_method cuda_event. Any of these present
# means a two-column run was intended, in which case ALL of them are required (a partial schema would let the gates below skip silently).
TWO_COLUMN_REQUIRED = {"time_stats.forward_gpu_span.median", "time_stats.forward_gpu_span.count", "host_wall_per_forward_ms",
                       "host_wall_per_forward_ms_backlog", "gpu_backlog_ms", "gpu_backlog_ms_actual",
                       *(f"time_stats_hostbound.{op}.median" for op in ATTN_OPS), *(f"time_stats_hostbound.{op}.count" for op in ATTN_OPS),
                       *(f"legacy_host_bound_ratio.{op}" for op in ATTN_OPS), *(f"legacy_host_bound.{op}" for op in ATTN_OPS)}
TWO_COLUMN_SIGNATURE = ("time_stats_hostbound.", "legacy_host_bound", "gpu_backlog_ms", "host_wall_per_forward_ms_backlog", "time_stats.forward_gpu_span.")
# Second all-or-nothing group (runs from 2026-09-17 10:16 on): concurrent shader-clock estimate (MHz) immediately before/after each timed
# loop and the host enqueue time per pass. Required only when any of them is present, so the pre-probe two-column run
# (runs/2026-09-17_0846_...) stays checkable; its absence is reported in the notes. Non-cuda_event runs emit only the legacy pair as
# numbers (backlog ones NaN) and are legacy-schema files, which is why only *_backlog names may act as signatures.
PROBE_REQUIRED = {"sclk_mhz_legacy_start", "sclk_mhz_legacy_end", "sclk_mhz_backlog_start", "sclk_mhz_backlog_end",
                  "host_enqueue_per_forward_ms", "host_enqueue_per_forward_ms_backlog"}
ok = lambda cond, msg: None if cond else fails.append(msg)
def same_rows(a, b):  # same multiset of rows, order-insensitive, NaN-safe
    if len(a) != len(b) or set(a.columns) != set(b.columns): return False
    h = lambda df: np.sort(pd.util.hash_pandas_object(df.reindex(columns=sorted(df.columns)), index=False).to_numpy())
    return np.array_equal(h(a), h(b))
OPS = ("attn_kv_cache_save", "attn_prefill", "attn_decode")
BATCH = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512]
DK = [v for v in (128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536) if v <= MSL]
TM_CHUNKS = [c for c in (1024, 4096, 8192, 16384, 32768) if c < MSL]
def token_budget(tp, block):  # what the profiler's memory filter allows: utils.get_max_num_blocks * block_size, with the MI355X total (309,220,868,096 B)
    return int(0.9 * 309220868096 // (2 * block * max(1, 4 // tp) * 128 * 2 * 48)) * block
REQ = {"n_embd", "n_q_head", "n_kv_head", "head_dim", "block_size", "num_tensor_parallel_workers", "max_model_len", "batch_size",
       "prefill_chunk_size", "kv_cache_size", "is_prefill", "attention_backend", "profiling_precision", "measurement_type",
       "warmup_steps", "active_steps", *(f"time_stats.{op}.{s}" for op in OPS for s in ("median", "count", "samples"))}

ok(not [f for f in set(d.glob("*.csv")) | set(cells.glob("*.csv")) if f.is_symlink()], "symlinked csv in dataset dir")
read = lambda f: pd.read_csv(f, low_memory=False, float_precision="round_trip")  # exact floats: the row-hash checks compare re-written CSVs
if (d / "attention.csv").exists():
    att = {n: read(d / f"{n}.csv") for n in ("attention", "attention_true_mixed", "attention_combined")}
    for n, df in att.items():
        missing = sorted(REQ - set(df.columns))
        if missing:
            fails.append(f"{n}: missing columns {missing}"); continue
        ok(set(df.head_dim) == {128} and set(df.n_q_head) == {32} and set(df.n_kv_head) == {4} and set(df.n_embd) == {2048}, f"{n}: model dims wrong")
        ok(set(df.attention_backend) == {"AITER"}, f"{n}: attention_backend {sorted(set(df.attention_backend))} != AITER")
        ok(set(df.block_size) == {1, 16} and set(df.num_tensor_parallel_workers) == {1, 2, 4, 8}, f"{n}: block/TP sets {sorted(set(df.block_size))} {sorted(set(df.num_tensor_parallel_workers))}")
        ok(set(df.max_model_len) == {MSL} and set(df.profiling_precision) == {"BF16"} and set(df.measurement_type) == {"CUDA_EVENT"}, f"{n}: max_model_len/precision/measurement")
        ok(set(df.warmup_steps) == {3} and set(df.active_steps) == {50}, f"{n}: warmup/active steps")
        for op in OPS:
            ok(set(df[f"time_stats.{op}.count"]) == {50}, f"{n}: {op}.count != 50")
            ok(df[f"time_stats.{op}.samples"].map(lambda s: len(json.loads(s))).eq(53).all(), f"{n}: {op}.samples not 53 long")
        pre = df.is_prefill.astype(bool)
        ok(df[pre]["time_stats.attn_prefill.median"].notna().all() and df[~pre]["time_stats.attn_decode.median"].notna().all()
           and df["time_stats.attn_kv_cache_save.median"].notna().all(), f"{n}: NaN medians")
        ok(same_rows(df, pd.concat([read(cells / f"{n}_aiter_block{b}.csv") for b in (1, 16)])), f"{n}: canonical content != block1 + block16 (rebuild it with float_precision='round_trip')")
    cat = pd.concat([att["attention"], att["attention_true_mixed"]])
    for c in ("is_true_mixed_batch", "is_mixed_batch"):  # the profiler default-fills these markers in the combined file only
        if c in cat.columns: cat[c] = cat[c].fillna(False).astype(bool)
    ok(same_rows(att["attention_combined"], cat), "combined != attention + true_mixed")

    tm_cols = {"num_prefill_seqs", "prefill_seq_lens", "decode_batch_size", "decode_kv_cache_sizes", "decode_avg_kv_cache_size"}
    missing_tm = sorted(tm_cols - set(att["attention_true_mixed"].columns))
    ok(not missing_tm, f"attention_true_mixed: missing columns {missing_tm}")

    std = att["attention"]
    if not fails:
        key = ["block_size", "num_tensor_parallel_workers", "batch_size", "prefill_chunk_size", "kv_cache_size", "is_prefill"]
        for op in OPS:  # slowest-run position: a spike at 0-2 means warm-up leaks past 3 warm-ups
            pos = std[f"time_stats.{op}.samples"].map(lambda s: max(range(len(json.loads(s))), key=json.loads(s).__getitem__))
            notes.append(f"{op}: argmax run position histogram (0-2 = warm-up) {pos.value_counts().sort_index().to_dict()}")
            ratio = std[f"time_stats.{op}.samples"].map(lambda s: max(json.loads(s)[3:]) / (sorted(json.loads(s)[3:])[25] or 1))
            notes.append(f"{op}: max/median ratio over timed runs: p50 {ratio.quantile(.5):.2f} p95 {ratio.quantile(.95):.2f} max {ratio.max():.2f}")
        dup = std.groupby(key)["time_stats.attn_prefill.median"].agg(["min", "max", "size"]); dup = dup[dup["size"] > 1]
        spread = (dup["max"] - dup["min"]) / dup["max"]
        ok(dup.empty or spread.median() < 0.05, f"repeated shapes disagree: median rel spread {spread.median():.3f} >= 5%")
        notes.append(f"repeated shapes: {len(dup)}, rel spread median {spread.median():.3f} max {spread.max():.3f}" if len(dup) else "no repeated shapes")
        dec = std[~std.is_prefill.astype(bool)]
        mono = dec.sort_values("kv_cache_size").groupby(["block_size", "num_tensor_parallel_workers", "batch_size"])["time_stats.attn_decode.median"].apply(lambda s: s.is_monotonic_increasing).mean()
        notes.append(f"decode median non-decreasing in kv for {mono:.0%} of (block,TP,batch) series")
        mono_b = dec.sort_values("batch_size").groupby(["block_size", "num_tensor_parallel_workers", "kv_cache_size"])["time_stats.attn_decode.median"].apply(lambda s: s.is_monotonic_increasing).mean()
        notes.append(f"decode median non-decreasing in batch for {mono_b:.0%} of (block,TP,kv) series")
        xb = dec[dec.block_size == 1].merge(dec[dec.block_size == 16], on=["num_tensor_parallel_workers", "batch_size", "kv_cache_size"], suffixes=("_b1", "_b16"))
        xr = xb["time_stats.attn_decode.median_b1"] / xb["time_stats.attn_decode.median_b16"]
        notes.append(f"block1/block16 decode median ratio at the same shape: p50 {xr.quantile(.5):.2f} p5 {xr.quantile(.05):.2f} p95 {xr.quantile(.95):.2f}")
        tm = att["attention_true_mixed"]
        exp_tm = {(len(a.prefill_seq_lens), a.prefill_seq_lens[0], a.decode_batch_size, a.decode_kv_cache_sizes[0]) for a in get_true_mixed_attention_input_combinations(
            max_seq_len=MSL, prefill_batch_sizes=[1, 2], prefill_chunk_sizes=TM_CHUNKS, decode_batch_sizes=BATCH, decode_kv_cache_sizes=DK, prefill_kv_cache_size=0)}
        if not missing_tm:
            tmd = tm[tm.num_prefill_seqs == 1].merge(dec, left_on=["block_size", "num_tensor_parallel_workers", "decode_batch_size", "decode_avg_kv_cache_size"],
                                                     right_on=["block_size", "num_tensor_parallel_workers", "batch_size", "kv_cache_size"], suffixes=("_tm", "_even"))
            if len(tmd):
                ov = tmd["time_stats.attn_decode.median_tm"] / tmd["time_stats.attn_decode.median_even"] - 1
                # informational: the true-mixed attn_decode scope times the decode part of a mixed batch, the even one a pure decode batch;
                # on gpt-oss AITER data this reads about -50%, so it is a consistency signal, not a like-for-like overhead
                notes.append(f"true-mixed vs even decode at the same (TP, batch, kv): median ratio-1 {ov.median():+.1%}, p95 {ov.quantile(.95):+.1%} ({len(tmd)} pairs)")
            for b, tp in [(b, tp) for b in (1, 16) for tp in (1, 2, 4, 8)]:  # explicit grid: a wholly missing cell must fail
                g = tm[(tm.block_size == b) & (tm.num_tensor_parallel_workers == tp)]
                got = {(int(n), json.loads(p)[0], int(db), json.loads(k)[0]) for n, p, db, k in zip(g.num_prefill_seqs, g.prefill_seq_lens, g.decode_batch_size, g.decode_kv_cache_sizes)}
                ok(got <= exp_tm, f"block{b} TP{tp}: {len(got - exp_tm)} unplanned true-mixed shapes")
                unexplained = [k for k in exp_tm - got if k[0] * k[1] + k[2] * (1 + k[3]) < 0.95 * token_budget(tp, b)]
                ok(not unexplained, f"block{b} TP{tp}: {len(unexplained)} missing true-mixed shapes (not explained by the KV memory budget)")
                ok((1, 1024, 1, 128) in got, f"block{b} TP{tp}: smallest true-mixed shape missing (memory filter cannot explain)")
                notes.append(f"block{b} TP{tp}: {len(g)}/{len(exp_tm)} true-mixed rows, {len(exp_tm - got)} memory-filtered")
        combos = get_attention_input_combinations(MSL, 1, 512, False, False, BATCH, DK, True, -1)
        exp = {(a.prefill_chunk_size, a.kv_cache_size, a.batch_size, a.is_prefill) for a in combos}  # 2772 attempts, 2497 unique keys
        for b, tp in [(b, tp) for b in (1, 16) for tp in (1, 2, 4, 8)]:  # explicit grid: a wholly missing cell must fail
            g = std[(std.block_size == b) & (std.num_tensor_parallel_workers == tp)]
            got = set(zip(g.prefill_chunk_size, g.kv_cache_size, g.batch_size, g.is_prefill.astype(bool)))
            ok(got <= exp, f"block{b} TP{tp}: {len(got - exp)} unplanned shapes {sorted(got - exp)[:3]}")
            miss = exp - got
            ok(all(not m[3] for m in miss), f"block{b} TP{tp}: missing prefill shapes {sorted(m for m in miss if m[3])[:5]}")
            unexplained = [m for m in miss if m[2] * (m[0] + m[1]) < 0.95 * token_budget(tp, b)]
            ok(not unexplained, f"block{b} TP{tp}: {len(unexplained)} missing decode shapes (not explained by the KV memory budget) {sorted(unexplained)[:3]}")
            ok((0, 128, 1, False) in got, f"block{b} TP{tp}: smallest decode shape missing (memory filter cannot explain)")
            notes.append(f"block{b} TP{tp}: {len(g)}/{len(combos)} standard rows, {len(miss)} memory-filtered decode shapes")
else:
    notes.append("no attention.csv: attention checks skipped")

if (d / "linear_op.csv").exists():
    lin = read(d / "linear_op.csv")
    tokens = TOKENS_GRID if TOKENS_GRID is not None else set(get_num_tokens_to_profile(16384)) - {4000}
    ok(set(lin.num_tensor_parallel_workers) == {1, 2, 4, 8} and set(lin.n_head) == {32} and set(lin.n_kv_head) == {4} and set(lin.n_embd) == {2048} and lin.use_qk_norm.all(), "linear_op: dims/TP/qk_norm")
    ok(lin["time_stats.attn_pre_proj.median"].notna().all() and lin["time_stats.attn_post_proj.median"].notna().all(), "linear_op: NaN in attn_pre/post_proj")
    ok(not [c for c in lin.columns if c.startswith("time_stats.add.")], "linear_op: add scope present (should be fused into RMSNorm)")
    ok(tokens <= set(lin.num_tokens) and len(lin) == lin.num_tokens.nunique() * 4, f"linear_op: {len(lin)} rows, {lin.num_tokens.nunique()} token values (need the {len(tokens)}-value base grid at every TP; missing {sorted(tokens - set(lin.num_tokens))[:5]})")
    notes.append(f"linear_op: {lin.num_tokens.nunique()} token values x 4 TP = {len(lin)} rows")
    ok(lin[lin.num_tensor_parallel_workers > 1]["time_stats.emb.median"].isna().all() and lin[lin.num_tensor_parallel_workers == 1]["time_stats.emb.median"].notna().all(),
       "linear_op: emb must be recorded on TP=1 rows only (replicated ops are split to TP=1)")
    # two-column timing schema (GPU-bound time_stats.* + legacy time_stats_hostbound.*) and its hard gates
    # a signature column counts only if it carries a value: non-cuda_event runs of the same wrapper write the four backlog/host-wall
    # scalars as all-NaN headers (their dict-valued fields expand to no columns), and those runs are legacy-schema by design
    two_col_intended = any(c.startswith(TWO_COLUMN_SIGNATURE) and lin[c].notna().any() for c in lin.columns)
    missing_two_col = sorted(TWO_COLUMN_REQUIRED - set(lin.columns)) if two_col_intended else []
    if two_col_intended and missing_two_col:
        ok(False, f"linear_op: two-column timing schema incomplete - missing {len(missing_two_col)} required column(s), gates not run: {missing_two_col[:6]}")
    elif two_col_intended:
        active = int(lin.active_steps.iloc[0]) if "active_steps" in lin else 50
        cover = lin.gpu_backlog_ms_actual >= BACKLOG_COVERAGE_FACTOR * active * lin.host_wall_per_forward_ms
        ok(cover.all(), f"linear_op: GPU backlog covers < {BACKLOG_COVERAGE_FACTOR}x the legacy loop on {int((~cover).sum())} rows (min ratio {(lin.gpu_backlog_ms_actual / (active * lin.host_wall_per_forward_ms)).min():.2f})")
        for op in GEMM_SCOPES:  # the mlp scopes are optional by design: --is_moe drops them (main.filter_mlp_columns)
            g, l = f"time_stats.{op}.median", f"time_stats_hostbound.{op}.median"
            if g in lin and l in lin:
                r = (lin[g] / lin[l]).dropna()
                ok((r <= GPU_BOUND_MAX_OVER_LEGACY).all(), f"linear_op: GPU-bound {op} exceeds {GPU_BOUND_MAX_OVER_LEGACY}x legacy on {int((r > GPU_BOUND_MAX_OVER_LEGACY).sum())} rows (max {r.max():.3f})")
        flags = {c.split(".", 1)[1]: int(lin[c].eq(True).sum()) for c in lin.columns if c.startswith("legacy_host_bound.")}  # NaN (replicated ops at TP>1) is not a flag
        notes.append(f"linear_op: two-column schema; legacy column host-bound (ratio > 1.15) on rows per op: {flags} (informational, not a failure)")
        probe_present = PROBE_REQUIRED & set(lin.columns)
        if probe_present:
            missing_probe = sorted(PROBE_REQUIRED - set(lin.columns))
            ok(not missing_probe, f"linear_op: clock-probe columns incomplete - missing {missing_probe}")
            if not missing_probe:
                bad = {c: int((~np.isfinite(lin[c].astype(float))).sum()) for c in sorted(PROBE_REQUIRED) if not np.isfinite(lin[c].astype(float)).all()}
                ok(not bad, f"linear_op: clock-probe/enqueue columns with non-finite values (rows per column): {bad}")
                clk = {c: (float(lin[c].median()), float(lin[c].min()), float(lin[c].max())) for c in sorted(PROBE_REQUIRED) if c.startswith("sclk_mhz_")}
                notes.append("linear_op: concurrent SCLK estimate (MHz) median [min-max]: " + ", ".join(f"{k.replace('sclk_mhz_', '')}={v[0]:.0f} [{v[1]:.0f}-{v[2]:.0f}]" for k, v in clk.items()))
        else:
            notes.append("linear_op: two-column run WITHOUT the clock probe (pre-2026-09-17 10:16); clock state of the two passes unknown")
    else:
        ok(ALLOW_LEGACY, "linear_op: single-column (legacy) timing schema - cuda_event runs before 2026-09-17 are host-bound below ~5k tokens at TP>1 (07_ §3.6); kineto/perf_counter/record_function runs have no second pass by design; pass --allow-legacy-schema to accept")
        if ALLOW_LEGACY: notes.append("linear_op: legacy single-column schema accepted (--allow-legacy-schema)")
else:
    notes.append("no linear_op.csv: linear checks skipped")

if not (d / "attention.csv").exists() and not (d / "linear_op.csv").exists():
    fails.append(f"nothing checked - neither attention.csv nor linear_op.csv in {d}")
print("\n".join(notes)); print("\n".join("FAIL: " + f for f in fails) or "PASS"); sys.exit(1 if fails else 0)
