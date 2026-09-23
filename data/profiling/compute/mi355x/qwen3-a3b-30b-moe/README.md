# Qwen3-30B-A3B on MI355X — attention and attention-layer linear-op profiling data

Collected 2026-09-15 on the Memphis Slurm cluster (partition `XAI`, nodes `amd-mi355x-8` / `-9`, 8× AMD MI355X gfx950 per
node) with Frontier's operator profilers. Model config: `data/config/models/qwen3-a3b-30b-moe.json` (32 query heads, 4 KV heads,
head_dim 128, hidden 2048, 48 layers, BF16). Plan, run record and sanity script live in
`profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/` (`01_plan.md`, `02_run_record.md`, `sanity_check.py`).

## 1. Layout — canonical files vs runs

**Where the data lives (convention from 2026-09-22).** Large CSVs are not committed. They live on the shared filesystem under
`/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/<linear_op|attention>/<run-name>/`, each with a `README.md`
(copy of the run's RUN.md) and `SHA256SUMS`. A run dir here that holds only a `RUN.md` is a pointer to that location. Files still
tracked in git (the root trio, the 16k attention run, the small validation grids) predate the convention and are left as they are.

```
qwen3-a3b-30b-moe/
├── attention.csv, attention_true_mixed.csv, attention_combined.csv   canonical AITER attention data (16k context)
├── linear_op.csv                                                     canonical linear-op data (dense 3,327-token grid)
├── README.md                                                         this file
└── runs/<YYYY-MM-DD_HHMM>_<what>/                                     one folder per profiler run, each with a RUN.md
```

**The root CSVs are the data to use.** Each is a verbatim copy (or, for attention, the block-1 + block-16 union) of the files
in one run folder; the run folder's `RUN.md` says when/where/how it was collected and what differed from the previous run.
Never analyse a run folder without reading its `RUN.md` first: some folders are diagnostics, not full grids.

| Run folder | Content | Status |
|---|---|---|
| `runs/legacy_pre-2026-09_torch-sdpa/` | TORCH_SDPA attention (block 16, 9.4k ctx, 5 reps), linear_op ≤4096 tokens (20 reps), `moe.csv` | **old — do not use** |
| `runs/2026-09-15_1142_attention_16k/` | AITER attention, 16k, blocks 1 and 16, per-block files | **canonical**: root `attention*.csv` = block1 ∪ block16 |
| `runs/2026-09-15_1142_linear_op_grid386/` | linear ops, 386-token default grid | superseded by the dense run; kept for comparison |
| `runs/2026-09-15_1308_linear_op_dense3327/` (pointer only; CSV on `/opt/shared/frontier-qwen3-profiling/datasets/…/linear_op/…`) | linear ops, dense 3,327-token grid | root `linear_op.csv`; **superseded 2026-09-22** (host-bound at TP>1 below ~5k tokens, `attn_rope` = wrong torch fallback, GC spike on 32 rows — `12_dip_investigation_summary.md`); kept unchanged as the pre-fix record |
| `runs/2026-09-15_1315_attention_32k/` (pointer only; CSVs on `/opt/shared/frontier-qwen3-profiling/datasets/…/attention/…`) | AITER attention, 32k, blocks 1 and 16, per-block files + union trio | validated (sanity PASS); not merged into root |
| `runs/2026-09-15_1315_attention_64k/` (pointer only; CSVs on `/opt/shared/frontier-qwen3-profiling/datasets/…/attention/…`) | AITER attention, 64k, blocks 1 and 16, per-block files + union trio | validated (sanity PASS); not merged into root |
| `runs/2026-09-15_1359_linear_op_spike_repro/` | 26 token values around the spike, 3 runs | diagnostic only |
| `runs/2026-09-16_0629_linear_op_dense3327_rerun/` | dense grid again, same command, node 8 | validated (sanity PASS); **spike reproduced exactly** → independent replicate, root file unchanged |
| `runs/2026-09-17_0846_linear_op_validation_grid_two_column/` | linear ops, 8-token validation grid × TP {1,2,4,8}, **two timing columns** (legacy + GPU-bound), see §5b | validated (`sanity_check.py --tokens-grid` PASS); first two-column output; **no clock probe** — superseded by the 10:16 run |
| `runs/2026-09-17_1016_linear_op_validation_grid_two_column_probe/` | same grid + 512 tokens, two timing columns **with the per-row clock probe and host-enqueue columns** (§5b) | validated (sanity PASS, `test_measurement_validity.py --expect green`); **reference validation-grid file**; the legacy pass starts its timed loop at ≈0.8 GHz on 32/36 rows *even after the 3 warm-up forwards* and ramps to ≈2.4 GHz during the loop, the GPU-bound pass runs ≈2.4 GHz throughout (RUN.md) — the root `linear_op.csv` is still single-column and host-bound below ~5k tokens at TP>1 |
| `runs/2026-09-17_1250_linear_op_rope_fix_validation_grid/` | validation grid, **attn_rope on the fused `rotary_embedding` kernel** (`attn_rope_impl=vllm_kernel`), cuda_event two-column file + record_function `linear_op_kernel_only.csv` | validated (sanity PASS incl. rope gate, validity GREEN); first run without the forced torch fallback: rope is 2–8× cheaper than every earlier `attn_rope` value; TASK-1 acceptance of the recollection plan |
| `runs/2026-09-22_0849_linear_op_dense_200fwd_all_fixes/` (pointer only; CSV on `/opt/shared/frontier-qwen3-profiling/datasets/…`) | **dense grid, all measurement fixes, 200 forwards as 8 spun blocks of 25** — GPU-bound timing, clock probe, GC guard, fused RoPE | validated (sanity PASS at gate 1.10, validity GREEN); **canonical for training from 2026-09-22**; the root `linear_op.csv` stays as the pre-fix record |

Adding a run: create `runs/<date>_<HHMM>_<what>/`, drop the profiler's CSVs in, write `RUN.md` (when, jobs, node, image,
commit, command, ops × shapes × repetitions table, "what differs from the previous run and why" with links, results),
then update the table above and, if the run becomes canonical, replace the root file and say so in both places.

## 2. New data vs old data — how to tell

Only the **new** generation (AITER, 2026-09-15 onwards) should be used for analysis.

| | NEW (2026-09-15 →) | OLD (`runs/legacy_pre-2026-09_torch-sdpa/`) |
|---|---|---|
| Attention backend | `attention_backend == "AITER"` (production kernels) | `TORCH_SDPA` (portable reference) |
| Columns present | `head_dim`, `warmup_steps`, `active_steps`, `time_stats.<op>.warmup_count`, `time_stats.<op>.samples` | none of these |
| Timed runs per shape | `time_stats.<op>.count == 50` | 5 (attention) / 20 (linear_op) |
| Block sizes | 1 **and** 16 | 16 only |
| Context ceiling | `max_model_len` 16384 (root), 32768 / 65536 (run folders) | 9472 (attention), 4096 tokens (linear_op) |

Rule of thumb in code: `"head_dim" in df.columns and set(df.attention_backend) == {"AITER"}` ⇒ new attention data;
`"warmup_steps" in df.columns` ⇒ new linear_op data. `moe.csv` is old and MoE was **not** re-collected (deferred).

## 2b. File inventory (root and the 16k run)

| File | Rows | What it holds |
|---|---|---|
| `attention.csv` | 22,152 | canonical: union of the two `attention_aiter_block*.csv` of the 16k run |
| `attention_true_mixed.csv` | 2,880 | canonical: union of the two true-mixed files |
| `attention_combined.csv` | 25,032 | canonical: standard + true-mixed — **this is what the simulator/regressor reads** |
| `linear_op.csv` | 13,308 | attention-layer GEMMs/norms/embedding vs `num_tokens` (dense 3,327-value grid) |
| `runs/2026-09-15_1142_attention_16k/attention_aiter_block{1,16}.csv` | 11,076 each | standard rows for one block size (1 = SGLang page size, 16 = vLLM default) |
| `…/attention_true_mixed_aiter_block{1,16}.csv` | 1,440 each | true-mixed rows for one block size |
| `…/attention_combined_aiter_block{1,16}.csv` | 12,516 each | standard + true-mixed for one block size |
| `runs/2026-09-15_1315_attention_32k/attention*.csv` | 21,979 std / 2,240 tm per block | same six per-block files plus the union trio, 32k context |
| `runs/2026-09-15_1315_attention_64k/attention*.csv` | 44,099 std / 3,190 tm per block | same, 64k context |

The canonical trio is byte-for-byte the concatenation of the per-block files (built with
`pd.read_csv(..., float_precision="round_trip")` then `pd.concat().to_csv()`); the per-block files are kept because each
was one independent profiler run. The simulator selects rows by exact match on `block_size`, so mixing both block sizes in
one file is intended. The 32k/64k trios are **not** merged into the root files: the root stays one context per file so the
`max_model_len`-dependent prefill grids do not overlap.

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

One row = one (`num_tokens`, TP) point for the attention-layer linear ops of one layer. `num_tokens` takes **3,327 values**:
every count 1…2048, every 8th from 2056 to 8192, every 16th from 8208 to 16384, **minus 4000** (faults the GPU on this stack);
this is a strict superset of the profiler's default 386-value grid, which the earlier run `runs/2026-09-15_1142_linear_op_grid386/linear_op.csv` used. TP ∈ {1, 2, 4, 8}. Ops: `attn_pre_proj` = fused QKV projection **including the QK-norm** (Qwen3 applies it inside this scope),
`attn_post_proj` = output projection, `attn_rope`, `input_layernorm`, `post_attention_layernorm`, `emb`. Same
`samples/warmup_count/count/…` scheme as attention (3 + 50 runs). **`emb`, `input_layernorm` and `post_attention_layernorm`
are recorded on TP = 1 rows only and are NaN on TP > 1 rows by design** (replicated ops are split to TP 1 by the profiler).
`emb` records two launches per forward (the profiler's model calls the embedding twice), so its `samples` list has 106 entries
(6 warm-up + 100 timed) — read `warmup_count` and `count` rather than assuming 53. There is no `time_stats.add.*`: the residual add
is fused into RMSNorm for this model. Constant columns (`use_qk_norm True`, `n_expanded_embd 768`, `vocab_size 151936`, …) can
be ignored.

## 5b. Two-column timing (runs from 2026-09-17 on) — and why the root `linear_op.csv` is only partly usable

The legacy CUDA-event loop enqueues 50 forwards with no synchronisation; whenever the GPU has drained everything queued before a scope,
the event pair measures the **host's launch span**, not the kernel (root cause and validation: `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/07_…md`,
`08_…md`, `09_…md`). In the root `linear_op.csv` that inflates `attn_pre_proj`/`attn_rope`/`attn_post_proj` medians 1.2–7× below roughly
5000–10000 tokens at TP 2/4/8 (and TP1 below ~2000–3000 tokens); above those bounds it agrees with kernel time within 1–3 %. The profiler
now times every shape twice under `--profile_method cuda_event`:

| Column | Meaning |
|---|---|
| `time_stats.<op>.*` | **GPU-bound pass**: a device spin is enqueued before the timed loop so the device runs behind the host; the event pair measures the scope's kernels plus one ≈3 µs dispatch gap per kernel boundary. Primary column. |
| `time_stats_hostbound.<op>.*` | **legacy pass**, unchanged loop: host launch span wherever the device was idle at scope start. Kept for comparison. |
| `time_stats.forward_gpu_span.*` | one event pair around each whole forward of the GPU-bound pass (closure check; see RUN.md "F6"). |
| `host_wall_per_forward_ms`, `host_wall_per_forward_ms_backlog` | host wall of the 50-forward loop / 50 incl. the trailing sync, legacy and GPU-bound pass (the latter contains the spin drain). |
| `gpu_backlog_ms`, `gpu_backlog_ms_actual` | requested and event-measured spin length (ms). |
| `legacy_host_bound_ratio.<op>`, `legacy_host_bound.<op>` | legacy/GPU-bound median and the `> 1.15` flag: marks the **legacy** column as host-bound on that row (informational). |
| `attn_rope_impl` | **which RoPE implementation `attn_rope` timed** (from the 2026-09-17 12:50 run on): `vllm_kernel` = Frontier's `RotaryEmbedding` calling the fused `vllm._custom_ops.rotary_embedding` kernel (one launch; the same kernel family SGLang's `RotaryEmbedding.forward_hip` launches on this image, timings identical to 0.1 µs); `torch_fallback` = the pure-torch path (forced by `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`; before the fix it rotated only the first 64 columns of head 0 — **every `attn_rope` value in runs before this column existed is that wrong computation**, 13 launch-bound kernels); `vllm_object_<method>` = vLLM's own object (on ROCm it dispatches to its 17-kernel torch path, 45–160 µs, never the fused op). `sanity_check.py` rejects anything but `vllm_kernel` unless `--allow-rope-fallback`. |
| `sclk_mhz_legacy_start/end`, `sclk_mhz_backlog_start/end` | **concurrent shader-clock estimate** (MHz): a fixed 2M-cycle `torch.cuda._sleep` spin timed by an event pair on the idle device immediately before each timed loop (after warm-up) and immediately after it. Rows from the 2026-09-17 10:16 run on carry it (the 08:46 run predates it and is accepted by the checker with a note). Caveat: the probe reads the DPM clock of a momentarily idle device. In the 10:16 run the legacy start probe — taken *after* the 3 warm-up forwards and a synchronize — read 648–948 MHz on 32/36 rows (the 1-token rows, which follow another task's activity, read ≈2.4 GHz), while legacy end and both GPU-bound probes read 2243–2424 MHz: three warm-up forwards do not bring the device up from its idle clock, so the legacy column's early timed samples run at ≈⅓ of the clock the GPU-bound column sees. Compare the two passes' probes before attributing a column difference to the instrument. |
| `host_enqueue_per_forward_ms`, `host_enqueue_per_forward_ms_backlog` | host wall from the first timed forward to the end of the last forward's *enqueue* (before the trailing synchronize) / 50 — the host-side pace, per pass; `host_wall_*` minus this is the time the host waited for the device. |

**The two columns also differ by clock state.** The GPU-bound pass runs after an 80–120 ms spin; the spin's own event-measured length shows
the shader clock at ≈2.0 GHz when started from idle (1.93–2.08 M cycles/ms, 96 processes over three jobs) and ≈2.4 GHz once warm
(2.38–2.40 M). A legacy/GPU-bound delta is therefore instrument change **plus** clock change; where the legacy loop was already GPU-bound
the columns agree within 1–2 %, which bounds the clock share for ≥ 20 µs GEMMs, while for ≤ 20 µs kernels the traced no-knob/knob ratio is
1.05–1.4. `attn_rope` is the numerically wrong torch fallback in both columns (RUN.md of the 2026-09-17 run). `sanity_check.py` requires the
full two-column schema for new runs (`--allow-legacy-schema` for older or non-`cuda_event` files).

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
- Validation: `python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/sanity_check.py <this dir> --cells runs/2026-09-15_1142_attention_16k`
  re-checks schema, grid coverage, row-hash equality of the canonical files against the per-block cells, and prints the
  warm-up-position histograms. Run folders are checked the same way (`sanity_check.py runs/<run> [--max-seq-len 32768|65536]`;
  attention or linear checks are skipped when that file is absent). PASS on the root, the 16k/32k/64k runs and both linear runs (2026-09-16).
