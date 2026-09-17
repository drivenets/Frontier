# Brief for a specialist: a reproducible mid-range speed-up (dip) in `attn_post_proj` and `attn_rope` at TP>1, tokens ≈4000–6000

**Status (2026-09-17): superseded by `07_post_proj_rope_dip_root_cause.md`.** The answer is a measurement artifact
(CUDA-event pairs measuring host launch overhead while the GPU is idle, not GPU kernel time), not a GPU/library effect —
see `07_` §1 and §5 ("what contradicts or complicates the brief") for exactly which claims below hold up and which don't.
`07_`'s conclusion is itself a pre-registered hypothesis awaiting independent validation (`08_measurement_validation_preregistration.md`,
Phases 1–4 not yet run as of this note). Kept here for the investigation record, not as the current understanding.

Prepared 2026-09-16. Goal: hand a ROCm / hipBLASLt / vLLM-parallel-layers specialist everything needed to explain **why**
this happens. Unlike `04_linear_op_spike_specialist_brief.md` (a rare ~200 ms stall on one run in 50), this is not noise:
it is a sustained shift in the **median** over a wide, contiguous `num_tokens` band, reproduced on two independent runs.

## 1. One-paragraph summary

Plotting `time_stats.<op>.median` against `num_tokens` (log-log; see `viz/linear_ops_run_groups_and_stats.ipynb`), every
linear op is expected to increase (near-)monotonically — more tokens, more GEMM/elementwise work. `attn_post_proj` and
`attn_rope`, at TP ∈ {2,4,8} only (not TP 1), instead show a clean **V-shaped dip**: the median *drops* by 17–37% relative
to the surrounding plateau somewhere in `num_tokens` ≈ 4000–6000, then climbs back sharply to (or above) the pre-dip level
within a few hundred tokens. `attn_pre_proj`, measured on the same GPUs in the same forward calls in the same token range,
shows no such V shape at any TP — it's a smooth rise with, at most, a gentle multi-thousand-token downward wobble (e.g. TP8:
a slow decline from ~0.087 ms at 3500 tokens to ~0.070 ms at 7000, no sharp trough or snap-back; see §4 note). The
post_proj/rope dip reproduces near-exactly in an independent run on a different node 17 h later (same op, same TP pattern,
trough within ~100 tokens, magnitude within ~2 points). We traced the profiler's code and found a clean, TP-dependent
structural difference between the two ops that dip sharply and the one that doesn't: `attn_post_proj`'s GEMM contraction
dimension `K` *changes with TP* (`K = 4096/TP`), while `attn_pre_proj`'s `K` is fixed at 2048 for every TP. We do not know
the mechanism, and we do not yet know if `attn_pre_proj`'s gentler wobble is a smaller instance of the same effect or unrelated
noise.

## 2. Environment

Same as `04_linear_op_spike_specialist_brief.md` §2 (same container image `v0.5.11-rocm700-mi35x`, same code, same sbatch).
Data used here: job 21313 (`runs/2026-09-15_1308_linear_op_dense3327/linear_op.csv`, node `amd-mi355x-9`) and job 21334
(`runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv`, node `amd-mi355x-8`, 17 h later) — the same two files
`04_`/`05_` used for the spike. No new data collection was needed for this brief; it is a re-analysis of the existing runs.

## 3. What exactly is measured (code-verified, this investigation)

`attn_post_proj` is `RowParallelLinear.apply_weights` only (`frontier/profiling/common/parallel_utils/tensor_parallel_layers.py:968`,
`_run_unquantized_gemm` → ROCm path → hipBLASLt, same dispatch family as `attn_pre_proj`). Critically, `o_proj` is
constructed with `reduce_results=False` at all three call sites in `linear_op_impl.py` (lines 197, 308, 478), so the
`if self.reduce_results and self.world_size > 1:` branch (line 988, which would do `reduce_from_tensor_model_parallel_region`
under a *separate* `_communication_timer`) is **never taken**. `attn_post_proj` is pure GEMM time; no NCCL/RCCL collective is
in the timed scope or anywhere in this profiler run. (This rules out an all-reduce-algorithm-boundary explanation, which
was our first hypothesis — worth stating explicitly so it isn't re-investigated.)

`attn_rope` (`linear_op_impl.py:543`) is `self.rotary_emb(positions, q, k)` — a RoPE kernel (`frontier.profiling.common.layers.rotary_embedding.get_rope`),
not a GEMM at all. It has no `_linear_timer`/hipBLASLt involvement.

Per-GPU GEMM shapes, `M = num_tokens`, both ops BF16, both dispatched through the same ROCm `rocm_unquantized_gemm_impl` →
hipBLASLt path as `attn_pre_proj` (see `04_` §3):

| op | K (per-GPU, contraction dim) | N (per-GPU, output dim) | K a function of TP? |
|---|---|---|---|
| `attn_pre_proj` (`qkv_proj`, `ColumnParallelLinear`) | 2048 (= `n_embd`, **not divided** — Column-parallel shards the output) | `(32·128 + 2·4·128)/TP` = 5120/2560/1280/640 for TP 1/2/4/8 | **No — fixed at 2048 for every TP** |
| `attn_post_proj` (`o_proj`, `RowParallelLinear`) | `32·128/TP` = 4096/2048/1024/512 for TP 1/2/4/8 (Row-parallel shards the input) | 2048 (= `n_embd`, not divided) | **Yes** |

`attn_rope`'s per-GPU problem size is `M = num_tokens` × per-GPU head count (`n_head/TP` for q, `n_kv_head/TP`-ish for k,
`head_dim` 128 fixed) — also TP-dependent, via a different (non-GEMM) kernel.

## 4. The observation, quantified

Method: `median = time_stats.<op>.median` (over the 50 timed runs, no filtering needed — this is a shift in the shape's own
median, not a within-shape single-run artifact; see §5). For each (op, TP): `plateau` = median of `median` over
`num_tokens` 2800–3200 (just before the dip), `trough` = min over 4000–6000, `recovered` = median of `median` over
7500–8200 (well after).

**Job 21313** (first run):

| op | TP | plateau (ms) | trough (ms) @ tokens | recovered (ms) | drop |
|---|---|---|---|---|---|
| `attn_post_proj` | 1 | 0.0511 | 0.0614 @ 4096 | 0.0945 | **−20%** (rises, no dip) |
| `attn_post_proj` | 2 | 0.0583 | 0.0380 @ 4192 | 0.0583 | **35%** |
| `attn_post_proj` | 4 | 0.0486 | 0.0477 @ 4064 | 0.0585 | 2% (negligible) |
| `attn_post_proj` | 8 | 0.0472 | 0.0432 @ 4056 | 0.0511 | 8% |
| `attn_rope` | 1 | 0.0693 | 0.0775 @ 4128 | 0.0994 | **−12%** (rises, no dip) |
| `attn_rope` | 2 | 0.0773 | 0.0637 @ 4160 | 0.0836 | 18% |
| `attn_rope` | 4 | 0.0929 | 0.0701 @ 5984 | 0.0673 | **25%** |
| `attn_rope` | 8 | 0.0917 | 0.0841 @ 5864 | 0.0828 | 8% |
| `attn_pre_proj` | 1 | 0.1467 | 0.1730 @ 4144 | 0.3293 | −18% (rises, no dip) |
| `attn_pre_proj` | 2 | 0.1089 | 0.1240 @ 4448 | 0.1848 | −14% (rises, no dip) |
| `attn_pre_proj` | 4 | 0.0862 | 0.0889 @ 4384 | 0.1014 | −3% (flat, no dip) |
| `attn_pre_proj` | 8 | 0.0854 | 0.0708 @ 5920 | 0.0689 | 17% — **but shape differs**: a slow multi-thousand-token decline (0.087→0.070 ms from tokens 3500→7000, fine-grained check), not the sharp V with snap-back seen in post_proj/rope (§4 note below) |

**Job 21334** (independent rerun, different node, 17 h later) — same pattern, same order of magnitude, same TPs affected,
troughs within ~100 tokens of the first run:

| op | TP | plateau | trough @ tokens | recovered | drop |
|---|---|---|---|---|---|
| `attn_post_proj` | 2 | 0.0584 | 0.0369 @ 4016 | 0.0589 | **37%** |
| `attn_post_proj` | 4 | 0.0479 | 0.0482 @ 4040 | 0.0549 | −1% |
| `attn_post_proj` | 8 | 0.0468 | 0.0428 @ 4080 | 0.0511 | 9% |
| `attn_rope` | 2 | 0.0710 | 0.0635 @ 4096 | 0.0839 | 11% |
| `attn_rope` | 4 | 0.0898 | 0.0652 @ 5912 | 0.0674 | **27%** |
| `attn_rope` | 8 | 0.0938 | 0.0873 @ 4080 | 0.0831 | 7% |
| TP1, both ops | — | — | — | — | still rises, no dip, both runs |

**Method caveat:** the plateau/trough/recovered numbers above are a fixed-window summary (min over 4000–6000 vs. two
reference windows) — it flags any low point in that range, whether it's a sharp V or a gentle slope. Always check the
*shape*, not just the percentage: `attn_post_proj`/`attn_rope` at TP 2/4 show an unmistakable sharp trough with a snap-back
recovery within a few hundred tokens (§8's fine-grained print, or the notebook plot); `attn_pre_proj` does not, at any TP
— its "17%" at TP8 is a slow multi-thousand-token decline with no snap-back (verified by printing every 6th point from
3500–7200; see the reproduction note in §8). Treat `attn_pre_proj` as *not* exhibiting the phenomenon this brief is about,
pending the specialist's own look.

Takeaways: **TP2 gives `attn_post_proj`'s cleanest, largest, most reproducible dip** (35% / 37%, trough within 176 tokens
between runs). **TP4 gives `attn_rope`'s cleanest, largest dip** (25% / 27%, trough within 72 tokens between runs). TP1 is a
clean negative control for both ops in both runs. TP8 shows a smaller dip in both ops, both runs. The two ops' troughs are
not at the same `num_tokens` (post_proj ≈4000–4200 across TPs; rope ≈4100–4200 at TP2 but ≈5900–6000 at TP4) — they may be
two related-but-distinct effects rather than one.

## 5. What we ruled out

- **Not a within-shape outlier.** Unlike the `attn_pre_proj` stall, this shift is in the row's own `median` (50 timed runs),
  not driven by one freak run pulling up a `mean`/`max`. A quick scan (`time_stats.<op>.std` in the dip region) shows normal
  variance, not an inflated one — i.e., most/all of the 50 runs in a trough row are genuinely fast, not one lucky run.
- **Not a communication/all-reduce effect.** `reduce_results=False` for `o_proj` in this profiler (§3) — no collective is in
  the timed scope for `attn_post_proj`, so an RCCL algorithm-boundary theory for the dip is not applicable to this data.
- **Not present in `attn_pre_proj`.** Same GPUs, same worker processes, same forward calls, immediately before/after
  `attn_post_proj`/`attn_rope` in program order — pre_proj does not dip at any TP. Rules out anything that would uniformly
  affect the whole forward (clock state, host stalls, allocator behavior common to all ops in the call) as a *sufficient*
  explanation, though it doesn't rule out an effect that only some kernels are sensitive to.
- **Reproducible, not a one-off.** Confirmed on an independent run, different node, 17 h later (§4). This is unlike the
  `attn_pre_proj` stall's *duration* (non-reproducible) but like its *position* (deterministic) — except here the
  "position" is a `num_tokens` band, not a fixed run index.

## 6. Candidate causes (our ranking — please confirm or refute)

1. **hipBLASLt kernel-selection boundary tied to `K`.** §3's table shows `attn_post_proj`'s per-GPU `K` (4096/TP) is the one
   dimension that changes with TP while `attn_pre_proj`'s `K` is TP-invariant — exactly tracking which op dips. hipBLASLt
   picks a kernel/tile config per (M, K, N, dtype); it is plausible that for `K ∈ {2048, 1024, 512}` (TP 2/4/8) a kernel is
   chosen whose latency vs. `M` is *not* monotonic (e.g., it is unusually well-suited to a specific M range — a genuine
   speed-up, not a slowdown elsewhere), while `K=4096` (TP1) picks a kernel with smooth M-scaling. Cheap test:
   `HIPBLASLT_LOG_LEVEL=5` / `HIPBLASLT_LOG_MASK` (as suggested in `04_` for a different purpose) around `num_tokens`
   3000/4200/6000/8000 at TP2, to see if the selected solution/kernel-ID actually changes at the trough boundaries.
2. **RoPE kernel occupancy/tiling boundary, independent mechanism.** `attn_rope` is not a GEMM, so candidate 1 doesn't
   directly apply, but its per-GPU problem size (`num_tokens × n_head_per_gpu × head_dim`) is also TP-dependent, and a
   custom elementwise/rotation kernel can have its own grid-occupancy sweet spots (e.g. a specific number of thread blocks
   per CU that only lines up for certain `num_tokens × n_head_per_gpu` products). The trough locations differing between
   `attn_post_proj` (TP2) and `attn_rope` (TP4) argues these may be *two* boundary phenomena in two different kernels
   that happen to land in a similar general vicinity, not one shared cause. Cheap test: vary `num_tokens` finely (every
   value, not the coarse dense grid) across 3800–6200 for `attn_rope` alone at TP2 and TP4 to map the trough shape
   precisely; check `rotary_embedding.py`'s kernel launch config (block/grid size formula) for a rounding/quantization step
   near those head-count × token products.
3. **A wall-clock/scheduling effect specific to TP>1 passes.** The `04_`/`05_` investigation showed the *pass* (all 3,327
   tokens, one TP) takes materially different wall time by TP (≈25–30 s), and other host-side events (GC) can and do land
   at fixed points in a pass. Against: a generic clock/thermal or scheduling effect should be TP-magnitude-independent in
   *location* (same wall-clock offset ⇒ different token, since cumulative work differs by TP) but the trough tokens here
   are *similar* across TP2/4/8 (~4000–6000) despite very different per-TP cumulative time to reach them — this argues
   against a pure wall-clock trigger and for something keyed to `num_tokens` (i.e., shape) itself. Still worth checking
   directly, using the same instrumentation `05_` already built (`spike_diag.py`) repurposed to log wall-time-per-forward
   across the whole pass rather than just around one index, to see if the dip aligns with a wall-time feature.
4. **Not applicable / lower priority:** all-reduce algorithm switching (ruled out, §5); Python GC (the `05_` root cause for
   the *other* anomaly — GC pauses make things *slower*, they don't explain a sustained *speed-up* band); shape-specific
   correctness bug (values are still physically reasonable GEMM/RoPE times, just faster than neighbors, and it reproduces
   — not obviously a numerics issue).

## 7. Questions we would like answered

1. Does the hipBLASLt-selected kernel/solution actually change at the `attn_post_proj` trough boundaries for TP2 (candidate 1)?
   If so, is the "fast" kernel actually a *better* choice that the library fails to pick outside the dip band (an
   opportunity, not a bug), or does it trade off something we aren't measuring (e.g., accuracy mode, workspace size)?
2. Is `attn_rope`'s dip mechanistically related to `attn_post_proj`'s, or coincidental? (The different trough locations
   by TP suggest maybe not — needs the fine-grained sweep in candidate 2 to settle.)
3. Why is TP2 the cleanest case for `attn_post_proj` and TP4 the cleanest for `attn_rope`, rather than the same TP for both?
4. Does this show up on other hardware the repo has data for (H100/H800/A800 `qwen3-a3b-30b-moe` runs also exist under
   `data/profiling/compute/{h100,h800,a800}/qwen3-a3b-30b-moe/linear_op.csv`, though likely a sparser grid — worth a quick
   check before assuming this is MI355X/hipBLASLt-specific)?

## 8. How to look at it yourself

- Rendered plots: `viz/linear_ops_run_groups_and_stats.ipynb`, `attn_post_proj`/`attn_rope` sections, "overall summary"
  (mean/median/envelope) plot per TP — the dip is visible directly on the log-scale median line for TP 2/4/8, absent for TP1.
- Re-derive the table in §4:
```python
import pandas as pd
D = "data/profiling/compute/mi355x/qwen3-a3b-30b-moe"
lin = pd.read_csv(f"{D}/linear_op.csv", low_memory=False, float_precision="round_trip")
for op in ["attn_post_proj", "attn_rope", "attn_pre_proj"]:
    col = f"time_stats.{op}.median"
    for tp in [1, 2, 4, 8]:
        d = lin[(lin.num_tensor_parallel_workers == tp) & lin[col].notna()].sort_values("num_tokens")
        plateau = d[(d.num_tokens.between(2800, 3200))][col].median()
        trough_region = d[d.num_tokens.between(4000, 6000)]
        trough = trough_region[col].min()
        trough_tok = trough_region.loc[trough_region[col].idxmin(), "num_tokens"]
        recovered = d[d.num_tokens.between(7500, 8200)][col].median()
        print(op, tp, plateau, trough, trough_tok, recovered, 100 * (plateau - trough) / plateau)
```
- Files: `frontier/profiling/linear_op/linear_op_impl.py` (scopes, `reduce_results=False` at lines 197/308/478),
  `frontier/profiling/common/parallel_utils/tensor_parallel_layers.py` (`RowParallelLinear.forward`/`apply_weights`,
  lines 807–999; `ColumnParallelLinear`, lines 621–806), `frontier/profiling/common/layers/rotary_embedding.py` (RoPE
  kernel). Data: `runs/2026-09-15_1308_linear_op_dense3327/linear_op.csv`, `runs/2026-09-16_0629_linear_op_dense3327_rerun/linear_op.csv`.
