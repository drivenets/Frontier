# Qwen3-30B-A3B on MI355X — attention and attention-layer linear-op profiling data

Collected 2026-09-15 on the Memphis Slurm cluster (partition `XAI`, nodes `amd-mi355x-8` / `-9`, 8× AMD MI355X gfx950 per
node) with Frontier's operator profilers. Model config: `data/config/models/qwen3-a3b-30b-moe.json` (32 query heads, 4 KV heads,
head_dim 128, hidden 2048, 48 layers, BF16). Plan, run record and sanity script live in
`profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/` (`01_plan.md`, `02_run_record.md`, `sanity_check.py`).

## 1. New data vs old data — how to tell

Two generations of data sit in this directory. Only the **new** generation should be used for analysis.

| | NEW (2026-09-15) | OLD (pre-2026-09, superseded) |
|---|---|---|
| Files | `attention*.csv` without a `_sdpa_` infix, `linear_op.csv` | `*_sdpa_block16.csv`, `linear_op_maxtokens4096.csv`, `moe.csv` |
| Attention backend | `attention_backend == "AITER"` (production kernels) | `TORCH_SDPA` (portable reference) |
| Columns present | `head_dim`, `warmup_steps`, `active_steps`, `time_stats.<op>.warmup_count`, `time_stats.<op>.samples` | none of these |
| Timed runs per shape | `time_stats.<op>.count == 50` | 5 (attention) / 20 (linear_op) |
| Block sizes | 1 **and** 16 | 16 only |
| Context ceiling | `max_model_len == 16384` | 9472 (attention), 4096 tokens (linear_op) |

Rule of thumb in code: `"head_dim" in df.columns and set(df.attention_backend) == {"AITER"}` ⇒ new attention data;
`"warmup_steps" in df.columns` ⇒ new linear_op data. `moe.csv` is old and MoE was **not** re-collected (deferred).

## 2. File inventory (new data)

| File | Rows | What it holds |
|---|---|---|
| `attention_aiter_block1.csv` | 11,076 | standard rows, `block_size == 1` (SGLang page size) |
| `attention_aiter_block16.csv` | 11,076 | standard rows, `block_size == 16` (vLLM default) |
| `attention_true_mixed_aiter_block1.csv` | 1,440 | true-mixed rows, block 1 |
| `attention_true_mixed_aiter_block16.csv` | 1,440 | true-mixed rows, block 16 |
| `attention_combined_aiter_block{1,16}.csv` | 12,516 each | standard + true-mixed for that block size |
| `attention.csv` | 22,152 | canonical: union of the two `attention_aiter_block*.csv` |
| `attention_true_mixed.csv` | 2,880 | canonical: union of the two true-mixed files |
| `attention_combined.csv` | 25,032 | canonical: union of the two combined files — **this is what the simulator/regressor reads** |
| `linear_op.csv` | 1,544 | attention-layer GEMMs/norms/embedding vs `num_tokens` |

The canonical trio is byte-for-byte the concatenation of the per-block files (built with
`pd.read_csv(..., float_precision="round_trip")` then `pd.concat().to_csv()`); the per-block files are kept because each
was one independent profiler run. The simulator selects rows by exact match on `block_size`, so mixing both block sizes in
one file is intended.

## 3. What one attention row is

One row = one **batch shape** measured on one GPU shard for **one transformer layer's attention call**. Constant columns:
`n_embd 2048`, `n_q_head 32`, `n_kv_head 4`, `head_dim 128`, `max_model_len 16384`, `attention_backend AITER`,
`profiling_precision BF16`, `measurement_type CUDA_EVENT`, `model_arch`/`model_architecture_profile` `generic`.

Shape columns:

- `block_size` ∈ {1, 16} — paged-KV page size.
- `num_tensor_parallel_workers` (TP) ∈ {1, 2, 4, 8} — the shard measured has 32/TP query heads and max(1, 4/TP) KV heads
  (at TP 8 each GPU holds one replicated KV head).
- `mode` — `even` for standard rows, `true_mixed` for mixed batches. `is_prefill` — bool (cast with `.astype(bool)`).

**Standard prefill rows** (`mode == "even"`, `is_prefill == True`): always `batch_size == 1`. `prefill_chunk_size` = new tokens
attended in this call; `kv_cache_size` = tokens already in the cache before the chunk. `kv_cache_size == 0` is a prefill from
scratch; `kv_cache_size > 0` is one chunk of a chunked prefill (its query attends to `kv_cache_size + prefill_chunk_size`
keys). Grid: chunk sizes 64 … 16384 (the profiler's `PREFILL_CHUNK_SIZE_SPACE`) at every chunk position
`kv_cache_size = k × chunk` below 16384, plus full prefills of every length in `get_seq_lengths_to_profile(16384)` at
`kv_cache_size 0`. Per (block, TP): 2,646 prefill attempts.

**Standard decode rows** (`is_prefill == False`): `batch_size` concurrent sequences each producing one token
(∈ {1,2,4,8,16,24,32,48,64,96,128,160,192,256,320,384,448,512}); `kv_cache_size` = context per sequence
(∈ {128,512,1024,2048,4096,8192,16384}); `prefill_chunk_size == 0`. Per (block, TP): 126 attempts, minus shapes the profiler's
KV-memory filter dropped: **9 at TP 1 and 3 at TP 2** (the largest batch × kv products), none at TP 4/8. Treat those as
"not measured", not as zero.

**True-mixed rows** (`mode == "true_mixed"`, `is_true_mixed_batch == True`): one batch containing `num_prefill_seqs` ∈ {1, 2}
fresh prefill requests of length `prefill_seq_lens` (JSON list; 1024, 4096 or 8192; `prefill_kv_cache_sizes` all 0) **and**
`decode_batch_size` decode requests, each at context `decode_avg_kv_cache_size` (`decode_kv_cache_sizes` is the JSON list,
all equal; values 128 … 8192). In these rows `batch_size == total_batch_size == num_prefill_seqs + decode_batch_size`
(≤ 128), `prefill_chunk_size` and `kv_cache_size` are 0 and **must not** be read as shape; use the true-mixed columns.
`time_stats.attn_prefill.*` times the prefill part of the mixed batch, `time_stats.attn_decode.*` the decode part.
360 shapes per (block, TP), all present.

Convenience columns on standard rows: `seq_lens` (JSON list of per-request token counts in this call), `total_tokens`,
`is_chunked_prefill_sample`, `chunk_start_token`, `chunk_end_token`, `total_prefill_tokens`.

**Duplicates are intentional.** Per (block, TP) the generator emits 2,772 prefill+decode attempts over 2,497 unique shapes:
269 shapes are measured twice and 3 three times (chunk sizes 128/1024/4096/16384 sit on two range seams, and a full prefill of
length L coincides with chunk L at kv 0). They are independent re-measurements; do not dedupe unless you want to.
Observed agreement: median relative spread 0.2 %, max ~20 %.

## 4. Timing columns — the part that matters for the warm-up / noise question

For each op `<op>` ∈ {`attn_prefill`, `attn_decode`, `attn_kv_cache_save`, `attn_input_reshape`, `attn_output_reshape`}
(attention) or {`attn_pre_proj`, `attn_post_proj`, `attn_rope`, `input_layernorm`, `post_attention_layernorm`, `emb`}
(linear_op), every row has:

| Column | Meaning |
|---|---|
| `time_stats.<op>.samples` | **JSON string of 53 floats, milliseconds, in execution order.** Runs 0–2 are the 3 warm-up runs, runs 3–52 the 50 timed runs. `json.loads` it. Values are rounded to 6 decimals. |
| `time_stats.<op>.warmup_count` | how many leading samples are warm-up (3 for every op in this dataset). |
| `time_stats.<op>.count` | number of timed runs (50). |
| `time_stats.<op>.min/max/mean/median/std` | computed over the **50 timed runs only** (warm-ups excluded). |
| `warmup_steps`, `active_steps` (row level) | 3 and 50: how the profiler was configured. |

Every op is timed on every row, so on a pure decode row `attn_prefill` samples are microsecond launch overhead (the kernel
is not called), and on a pure prefill row the same holds for `attn_decode`. Use `is_prefill`/`mode` to pick the meaningful op.
`attn_kv_cache_save` (scatter of the new K/V into the paged cache) is meaningful on every row. The two `*_reshape` ops are
trivial reshapes; ignore unless studying launch overhead.

Times are per **layer**; Qwen3-30B-A3B has 48 attention layers.

Already seen in this data: the slowest of the 53 runs is run 0 in ~88 % of `attn_prefill` rows and run 3 (first timed run
after the warm-up synchronize) in ~7 %; for `attn_decode` the slowest run is spread nearly uniformly with mild bumps at runs 0
and 48–49. So warm-up is a real, first-run effect that the 3 warm-ups absorb for the kernels, and the remaining spread is
noise-like. Any warm-up analysis must use `samples` (positions 0–2 vs 3–52), never the aggregates.

## 5. `linear_op.csv`

One row = one (`num_tokens`, TP) point for the attention-layer linear ops of one layer. `num_tokens` takes 386 values from 1 to
16384 (the profiler's `get_num_tokens_to_profile(16384)` grid **minus 4000**, which faults the GPU on this stack);
TP ∈ {1, 2, 4, 8}. Ops: `attn_pre_proj` = fused QKV projection **including the QK-norm** (Qwen3 applies it inside this scope),
`attn_post_proj` = output projection, `attn_rope`, `input_layernorm`, `post_attention_layernorm`, `emb`. Same
`samples/warmup_count/count/…` scheme as attention (3 + 50 runs). **`emb`, `input_layernorm` and `post_attention_layernorm`
are recorded on TP = 1 rows only and are NaN on TP > 1 rows by design** (replicated ops are split to TP 1 by the profiler).
`emb` records two launches per forward (the profiler's model calls the embedding twice), so its `samples` list has 106 entries
(6 warm-up + 100 timed) — read `warmup_count` and `count` rather than assuming 53. There is no `time_stats.add.*`: the residual add
is fused into RMSNorm for this model. Constant columns (`use_qk_norm True`, `n_expanded_embd 768`, `vocab_size 151936`, …) can
be ignored.

## 6. Reading recipes

```python
import json, pandas as pd
D = "data/profiling/compute/mi355x/qwen3-a3b-30b-moe"
att = pd.read_csv(f"{D}/attention_combined.csv", low_memory=False, float_precision="round_trip")
att["is_prefill"] = att.is_prefill.astype(bool)
std, tm = att[att["mode"] == "even"], att[att["mode"] == "true_mixed"]
pre, dec = std[std.is_prefill], std[~std.is_prefill]

# raw runs of one row, warm-ups first
runs = json.loads(dec.iloc[0]["time_stats.attn_decode.samples"]); warm, timed = runs[:3], runs[3:]

# decode time vs context for one (block, TP, batch)
s = dec[(dec.block_size == 1) & (dec.num_tensor_parallel_workers == 8) & (dec.batch_size == 64)]
s = s.sort_values("kv_cache_size")[["kv_cache_size", "time_stats.attn_decode.median"]]

# block 1 vs block 16 at the same shape
key = ["num_tensor_parallel_workers", "is_prefill", "batch_size", "prefill_chunk_size", "kv_cache_size"]
x = std[std.block_size == 1].merge(std[std.block_size == 16], on=key, suffixes=("_b1", "_b16"))

# true-mixed: expand the JSON columns
tm = tm.assign(prefill_len=tm.prefill_seq_lens.map(lambda s: json.loads(s)[0]))

lin = pd.read_csv(f"{D}/linear_op.csv", low_memory=False, float_precision="round_trip")
```

Use `float_precision="round_trip"` whenever you re-write CSVs that must stay comparable with these (pandas' default float
parser is not round-trip exact).

## 7. Provenance and caveats

- Attention was collected inside `lmsysorg/sglang:v0.5.17-rocm720-mi35x` (torch 2.9.1+rocm7.2.0). The originally pinned
  `v0.5.11-rocm700-mi35x` faults (GPU memory aperture violation) in aiter's `mha_batch_prefill` for head_dim 128 with paged KV,
  at both block sizes; see `02_run_record.md`. linear_op was collected inside `v0.5.11` (it needs vllm, which v0.5.17 dropped,
  and never calls the faulty kernel).
- Kernels: prefill `aiter.ops.mha.mha_batch_prefill_func`, decode `aiter.ops.attention.paged_attention_ragged`, called exactly as
  SGLang's `aiter_backend.py` calls them, but SGLang runs them at page size 1 only; block 16 is a kernel path production does not
  exercise (it ran clean here, but is not validated against SGLang output).
- Profiler: `frontier.profiling.attention.main` (4 GPU workers per cell) and `frontier.profiling.linear_op.main`; the exact command
  lines are in the Slurm logs under `logs/qwen3_mi355x/` on the collecting checkout and in `02_run_record.md`.
- Cold-start behaviour was **not** measured: every shape is preceded by its own 3 warm-up runs inside an already-hot process.
- Validation: `python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py <this dir>` re-checks schema, grid
  coverage, row-hash equality of the canonical files, and prints the warm-up-position histograms. It passed on this data.
