"""Sanity-check the collected Qwen3-30B-A3B MI355X dataset dir (plan Step 4). Usage: sanity_check.py <dataset dir>"""
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run by path without an editable install
from frontier.profiling.utils import get_attention_input_combinations, get_num_tokens_to_profile, get_true_mixed_attention_input_combinations

d = Path(sys.argv[1]); fails, notes = [], []
ok = lambda cond, msg: None if cond else fails.append(msg)
def same_rows(a, b):  # same multiset of rows, order-insensitive, NaN-safe
    if len(a) != len(b) or set(a.columns) != set(b.columns): return False
    h = lambda df: np.sort(pd.util.hash_pandas_object(df.reindex(columns=sorted(df.columns)), index=False).to_numpy())
    return np.array_equal(h(a), h(b))
OPS = ("attn_kv_cache_save", "attn_prefill", "attn_decode")
BATCH = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512]; DK = [128, 512, 1024, 2048, 4096, 8192, 16384]
REQ = {"n_embd", "n_q_head", "n_kv_head", "head_dim", "block_size", "num_tensor_parallel_workers", "max_model_len", "batch_size",
       "prefill_chunk_size", "kv_cache_size", "is_prefill", "attention_backend", "profiling_precision", "measurement_type",
       "warmup_steps", "active_steps", *(f"time_stats.{op}.{s}" for op in OPS for s in ("median", "count", "samples"))}

ok(not [f for f in d.glob("*.csv") if f.is_symlink()], "symlinked csv in dataset dir")
read = lambda f: pd.read_csv(f, low_memory=False, float_precision="round_trip")  # exact floats: the row-hash checks compare re-written CSVs
att = {n: read(d / f"{n}.csv") for n in ("attention", "attention_true_mixed", "attention_combined")}
for n, df in att.items():
    missing = sorted(REQ - set(df.columns))
    if missing:
        fails.append(f"{n}: missing columns {missing}"); continue
    ok(set(df.head_dim) == {128} and set(df.n_q_head) == {32} and set(df.n_kv_head) == {4} and set(df.n_embd) == {2048}, f"{n}: model dims wrong")
    ok(set(df.attention_backend) == {"AITER"}, f"{n}: attention_backend {sorted(set(df.attention_backend))} != AITER")
    ok(set(df.block_size) == {1, 16} and set(df.num_tensor_parallel_workers) == {1, 2, 4, 8}, f"{n}: block/TP sets {sorted(set(df.block_size))} {sorted(set(df.num_tensor_parallel_workers))}")
    ok(set(df.max_model_len) == {16384} and set(df.profiling_precision) == {"BF16"} and set(df.measurement_type) == {"CUDA_EVENT"}, f"{n}: max_model_len/precision/measurement")
    ok(set(df.warmup_steps) == {3} and set(df.active_steps) == {50}, f"{n}: warmup/active steps")
    for op in OPS:
        ok(set(df[f"time_stats.{op}.count"]) == {50}, f"{n}: {op}.count != 50")
        ok(df[f"time_stats.{op}.samples"].map(lambda s: len(json.loads(s))).eq(53).all(), f"{n}: {op}.samples not 53 long")
    pre = df.is_prefill.astype(bool)
    ok(df[pre]["time_stats.attn_prefill.median"].notna().all() and df[~pre]["time_stats.attn_decode.median"].notna().all()
       and df["time_stats.attn_kv_cache_save.median"].notna().all(), f"{n}: NaN medians")
    ok(same_rows(df, pd.concat([read(d / f"{n}_aiter_block{b}.csv") for b in (1, 16)])), f"{n}: canonical content != block1 + block16 (rebuild it with float_precision='round_trip')")
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
        max_seq_len=16384, prefill_batch_sizes=[1, 2], prefill_chunk_sizes=[1024, 4096, 8192], decode_batch_sizes=BATCH, decode_kv_cache_sizes=DK, prefill_kv_cache_size=0)}
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
            ok(not (exp_tm - got) or tp in (1, 2), f"block{b} TP{tp}: {len(exp_tm - got)} missing true-mixed shapes at TP>2")
            ok((1, 1024, 1, 128) in got, f"block{b} TP{tp}: smallest true-mixed shape missing (memory filter cannot explain)")
            notes.append(f"block{b} TP{tp}: {len(g)}/{len(exp_tm)} true-mixed rows, {len(exp_tm - got)} memory-filtered")
    combos = get_attention_input_combinations(16384, 1, 512, False, False, BATCH, DK, True, -1)
    exp = {(a.prefill_chunk_size, a.kv_cache_size, a.batch_size, a.is_prefill) for a in combos}  # 2772 attempts, 2497 unique keys
    for b, tp in [(b, tp) for b in (1, 16) for tp in (1, 2, 4, 8)]:  # explicit grid: a wholly missing cell must fail
        g = std[(std.block_size == b) & (std.num_tensor_parallel_workers == tp)]
        got = set(zip(g.prefill_chunk_size, g.kv_cache_size, g.batch_size, g.is_prefill.astype(bool)))
        ok(got <= exp, f"block{b} TP{tp}: {len(got - exp)} unplanned shapes {sorted(got - exp)[:3]}")
        miss = exp - got
        ok(all(not m[3] for m in miss), f"block{b} TP{tp}: missing prefill shapes {sorted(m for m in miss if m[3])[:5]}")
        ok(not miss or tp in (1, 2), f"block{b} TP{tp}: {len(miss)} missing decode shapes at TP>2 (memory filter cannot explain) {sorted(miss)[:3]}")
        ok((0, 128, 1, False) in got, f"block{b} TP{tp}: smallest decode shape missing (memory filter cannot explain)")
        notes.append(f"block{b} TP{tp}: {len(g)}/{len(combos)} standard rows, {len(miss)} memory-filtered decode shapes")

lin = read(d / "linear_op.csv")
tokens = set(get_num_tokens_to_profile(16384)) - {4000}
ok(set(lin.num_tensor_parallel_workers) == {1, 2, 4, 8} and set(lin.n_head) == {32} and set(lin.n_kv_head) == {4} and set(lin.n_embd) == {2048} and lin.use_qk_norm.all(), "linear_op: dims/TP/qk_norm")
ok(lin["time_stats.attn_pre_proj.median"].notna().all() and lin["time_stats.attn_post_proj.median"].notna().all(), "linear_op: NaN in attn_pre/post_proj")
ok(not [c for c in lin.columns if c.startswith("time_stats.add.")], "linear_op: add scope present (should be fused into RMSNorm)")
ok(set(lin.num_tokens) == tokens and len(lin) == len(tokens) * 4, f"linear_op: {len(lin)} rows, {lin.num_tokens.nunique()} token values (expected {len(tokens) * 4} / {len(tokens)})")
ok(lin[lin.num_tensor_parallel_workers > 1]["time_stats.emb.median"].isna().all() and lin[lin.num_tensor_parallel_workers == 1]["time_stats.emb.median"].notna().all(),
   "linear_op: emb must be recorded on TP=1 rows only (replicated ops are split to TP=1)")

print("\n".join(notes)); print("\n".join("FAIL: " + f for f in fails) or "PASS"); sys.exit(1 if fails else 0)
