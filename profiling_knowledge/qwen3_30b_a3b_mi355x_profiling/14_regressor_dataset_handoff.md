# Handoff: the Qwen3-30B-A3B / MI355X linear_op dataset for a new per-(op, TP) regressor

Written 2026-09-23 for the agent building the new trainer. Everything here is checked against the file itself; commands to
re-check are given at the end. Read this before touching the old trainer code (`frontier/training/*_trainer.py`), which is being
replaced and should not be treated as a specification.

## 1. Where the data is

**Use this file** (job 21539, 2026-09-23, both measurement fixes, 8 settle forwards):

```
/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv
    md5    3635e4d305ae5cf84884e275d3ff37ef      45,463,879 bytes, 13,308 rows x 150 columns
    SHA256SUMS, README.md (what/why/provenance/verification), linear_op_kernel_only.csv (see s6), slurm-21539.out
```
Same content as the collection's scratch copy `Frontier/data/profiling_dense_fixed_workbacklog_settle8/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv`
on the cluster checkout. Not yet marked canonical in `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/README.md` (that README still
points at the 200-forward run of job 21486, whose medians are 2-3 % higher because its clock was still ramping; see s5).
Do **not** use the root `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` in git: it is the pre-fix 2026-09-15 file
(host-bound at TP>1 below ~5k tokens, wrong RoPE kernel).

Local analysis environment: `/home/dn/.virtualenvs/qwen3-profiling` (numpy, pandas). Read with
`pd.read_csv(path, low_memory=False)`.

## 2. What one row is

One row = one (shape, tensor-parallel degree) cell of the profiler: the attention-layer linear operators of Qwen3-30B-A3B run in
isolation on one MI355X (gfx950) with dummy weights, BF16, for `num_tokens` tokens, with the layer sharded as it would be at
tensor-parallel degree `num_tensor_parallel_workers` (one GPU holds 1/TP of the heads; no collectives are timed). Each cell was
timed 25 times (3 warm-up + 8 untimed settle forwards + 25 timed forwards, all on a device that is held busy so the timings are
device time, not host launch time); the statistics over the 25 timed samples are the row's values.

Grid: `num_tokens` in {1..2048 step 1, 2056..8192 step 8 (4000 missing), 8208..16384 step 16} = 3,327 values, times TP in
{1, 2, 4, 8} = 13,308 rows. Every (num_tokens, TP) pair appears exactly once.

Model constants (identical on every row, keep for provenance, useless as features): `n_head` 32, `n_kv_head` 4, `n_embd` 2048,
`n_expanded_embd` 768 (per-expert; MoE experts are **not** in this file), `vocab_size` 151936, `use_qk_norm` True,
`use_gated_mlp` True, `attn_output_gate` False, head_dim 128 (not a column; from the model config). Node `amd-mi355x-1`, image
`lmsysorg/sglang:v0.5.11-rocm700-mi35x`, code commit `3a8eb55`.

## 3. Label

For each op the label is **`time_stats.<op>.median`**: the median over the 25 timed samples of the CUDA-event pair around the
op, in **milliseconds**, measured with the device running behind the host (kernel time + ~2.5-3 us of event-record cost per
scope boundary; `10_measurement_contract.md`). It is device time at the automatic boost clock (~2.4 GHz, recorded per row in
`sclk_mhz_backlog_start/end`), settled: the clock ramp that biased earlier files is removed to within 1 % (s5).

Ops and their coverage:

| op (`<op>` in the column name) | what is timed | TP rows | median range (ms) |
|---|---|---|---|
| `attn_pre_proj` | QKV GEMM (2048 -> (32*128 + 2*4*128)/TP) + q-norm + k-norm RMSNorm kernels and their reshapes | 1, 2, 4, 8 | 0.011 - 0.612 |
| `attn_post_proj` | o_proj GEMM (32*128/TP -> 2048), no all-reduce | 1, 2, 4, 8 | 0.006 - 0.188 |
| `attn_rope` | fused vLLM rotary-embedding kernel on q and k (`attn_rope_impl == "vllm_kernel"` on every row) | 1, 2, 4, 8 | 0.006 - 0.096 |
| `input_layernorm` | RMSNorm (fused add + norm) on the 2048-wide hidden state | 1 only (replicated op, identical at every TP) | 0.006 - 0.103 |
| `post_attention_layernorm` | same, after attention | 1 only | 0.006 - 0.112 |
| `emb` | embedding lookup (timed twice per forward: `count` = 50) | 1 only | 0.006 - 0.025 |
| `forward_gpu_span` | one event pair around the whole forward (all ops incl. untimed MLP scopes) | 1, 2, 4, 8 | 0.126 - 1.500 |

So a per-(op, TP) fit means 3 x 4 = 12 regressors for the attention ops plus 3 TP-1 regressors for the replicated ops
(15 in total; `forward_gpu_span` is a closure check, not an operator to model). The replicated ops' TP-1 rows apply to every
TP (the profiler runs them unsharded and the CSV writer drops the duplicate TP>1 rows).

Alternative label columns per op, same units: `mean`, `min`, `max`, `std`, `count`, `warmup_count`, and `samples` (JSON list of
all recorded samples **including** the warm-up ones: drop the first `warmup_count` entries to get the 25 timed samples in run
order). Use `median`. Do not use `mean`/`max`/`min` (single-sample events; s5), and do not use the `time_stats_hostbound.*`
columns (s4).

## 4. Features

The only variables in this file are:

| column | type | values | role |
|---|---|---|---|
| `num_tokens` | int | 1 - 16384, 3,327 values | **the feature** (the GEMM's M dimension = number of tokens in the forward) |
| `num_tensor_parallel_workers` | int | 1, 2, 4, 8 | the split key for the per-(op, TP) fits; changes the GEMM's N (pre_proj) or K (post_proj) as 1/TP |

Everything else is constant, or is instrument metadata that must not enter a model:
`host_wall_per_forward_ms*`, `host_enqueue_per_forward_ms*`, `gpu_backlog_ms*`, `sclk_mhz_*`, `legacy_host_bound*`,
`time_stats_hostbound.*` (the legacy single-pass timing kept as a record of the host-bound method: for ~60-80 % of the rows it
is the host's launch span, not device time), `warmup_steps`, `active_steps`, `settle_steps`, `backlog_kind`,
`measurement_type`, `profiling_precision`, `quant_signature`, `model_arch*`, `padded_*`, `share_*` (NaN).

If the regressor needs shape-derived inputs (FLOPs, bytes) build them from `num_tokens`, TP and the constants above:
pre_proj GEMM M = num_tokens, K = 2048, N = (4096 + 1024)/TP; post_proj M = num_tokens, K = 4096/TP, N = 2048; both BF16
weights (2 bytes/element); rope touches num_tokens x (32 + 4)/TP x 128 elements of q and k.

## 5. What the label looks like, and what to be careful about

- **Shape of the curves.** Every op is flat for small `num_tokens` (weight-read or fixed-cost bound: pre_proj is ~11-57 us
  from 1 to a few hundred tokens depending on TP) and then grows roughly linearly. There is no dip and no plateau in the
  mid range any more (those were artefacts of the old file).
- **Steps from hipBLASLt tile selection are real.** `attn_pre_proj`/`attn_post_proj` at TP 2/4/8 jump by 15-30 % at a
  handful of token counts (4352->4360, 4480->4488, 4864->4872, 5376->5384, ~11.8k-12.3k) where the GEMM library switches tile
  (07_ s3, 12_ s5). A smooth monotone model will misfit those neighbourhoods by that much; a piecewise or tree model will not.
  Do not "clean" them away. Small single steps also exist in the norms at 8192 and 14032 tokens.
- **Instrument floor.** Kernels below ~10 us (post_proj/rope/norms at small shapes, pre_proj at TP4/8 small shapes) carry a
  +-3-5 us floor from event-record cost and a ~20 % disagreement between tracers; treat sub-10 us labels as "about 6-10 us",
  not as precise. `10_measurement_contract.md` s3 says how hetcalc is meant to combine this with a separate launch-overhead term.
- **Noise level.** Per-row relative MAD of the 25 samples is ~0.55 % (median over rows); the row median is stable to well
  under 1 %. Two independent same-day collections agreed to a 1.000 median ratio (1st-99th percentile 0.96-1.04).
- **Residual position effect.** At TP1 and TP2 the 25 samples still drift down by ~1 % over the block (clock/thermal
  settling under full-node load, `13_` s8); the median is the mid-block value and is unaffected to <0.5 %. This is the only
  known remaining bias.
- **Differences from the previously "canonical" 200-forward file** (`.../linear_op/2026-09-22_0849_dense_200fwd_all_fixes/`,
  job 21486): that file's medians are 2-3 % higher (attention GEMMs 2.4 %, norms 2 %, rope 1.5 %) because they were taken
  mid-ramp; its per-run samples contain the two positional artefacts described in `13_`. If you must compare models fitted on
  both, expect that offset.
- **Pre-registered validity test.** `test_measurement_validity.py --expect green` passes every assertion except
  `T2 TP4 4096tok/64tok = 1.972` against a 2.0 threshold (2.04 on the older files); the settled clock made the 4096-token
  kernel 4 % faster while the 64-token kernel is fixed-cost bound. Known flake item, reported not re-thresholded.

## 6. The companion kernel-only file (optional cross-check)

`linear_op_kernel_only.csv` next to the main file is the same grid measured with `--profile_method record_function`: per op,
the **sum of kernel durations** from the PyTorch/kineto trace of one forward, no event-record cost, single sample per shape.
It should be ~2.5-3 us per scope boundary below the event-pair median (12 us for the five-kernel pre_proj scope). Use it only to
sanity-check a fitted model's intercept; do not train on it (1 sample, tracer-loaded device).

## 7. Re-checking the file

```
source /home/dn/.virtualenvs/qwen3-profiling/bin/activate
cd /home/dn/Frontier-qwen3-profiling
K=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling
python $K/sanity_check.py <dir containing linear_op.csv>                       # expected: PASS
python $K/test_measurement_validity.py <dir>/linear_op.csv --expect green      # expected: all PASS except T2 TP4 = 1.972
python $K/run_position/run_position_hist.py <dir>/linear_op.csv time_stats     # expected: argmin/argmax near 4 % per position (TP1/2: run 25 up to 17 %)
```
Documents, in reading order: `12_dip_investigation_summary.md` (what was wrong with the old file and how it was fixed),
`10_measurement_contract.md` (what the label means), `13_run_position_anomalies_root_cause.md` (the two remaining sample-level
artefacts and their fixes), the README next to the data (provenance and verification of this exact file).
