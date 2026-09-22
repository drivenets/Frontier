# The linear_op "dip": what we saw, what we suspected, what was actually wrong, how we fixed it, what changed

Qwen3-30B-A3B on MI355X, linear-op profiling for hetcalc. Written 2026-09-22 as a plain-language summary of documents `06_` to `11_`
and the implementation records in `amd-playground/.claude/debug-reports/`. Every number below is in one of those files with its job id.

## 1. What we saw in the initial run

The first dense collection (2026-09-15, job 21313; 3,327 token counts × TP 1/2/4/8, 13,308 rows) plotted `time_stats.<op>.median`
against `num_tokens`. Every op should cost more as tokens grow. Two ops did not:

- `attn_post_proj` and `attn_rope`, at TP 2, 4 and 8 only, showed a V-shaped **dip**: the median *fell* by 17–37 % relative to the
  surrounding plateau somewhere between 4,000 and 6,000 tokens, then climbed back within a few hundred tokens.
- `attn_pre_proj`, timed in the same forward calls on the same GPUs, had no such V.
- Below the dip the curves were flat plateaus for thousands of tokens, which also makes no sense for a GEMM whose M grows 100×.
- An independent rerun 17 hours later on another node (job 21334) reproduced the trough within about 100 tokens and 2 percentage points.
  So it was not noise.

A second, unrelated symptom sat in the same file: one row in ~50 had a single 150–290 ms sample inside `attn_pre_proj` (`04_`, `05_`).
That one was traced to CPython's cyclic garbage collector running between the start event and the kernel launch, and fixed by disabling
GC around the timed loop. It is mentioned here only because both symptoms lived in the same data and both are gone in the new runs.

## 2. What we suspected

The brief (`06_`) went looking for a GPU or library mechanism. Its two candidates:

1. **hipBLASLt kernel selection keyed to K.** `attn_post_proj`'s contraction dimension changes with TP (K = 4096/TP) while
   `attn_pre_proj`'s K is fixed at 2048; a tile switch at some M could make the post-projection GEMM genuinely faster past a boundary.
2. **A RoPE-kernel occupancy effect** that happened to sit in the same token band.

Both assume the number in the CSV is GPU time. The alternative, raised when the plateaus were looked at more closely, was that the
number is not GPU time at all: a CUDA-event pair records "when the device *reaches* the start event" to "when it reaches the end
event". If the device is idle when the start event is recorded, the pair measures the host's launch span for that scope, not the kernel.

## 3. How we verified it, including what we got wrong on the way

**Kernel traces vs event pairs (`07_`, jobs 21376–21378).** Running the same shapes under rocprofv3 and comparing kernel durations
with the event medians: at 8–64 tokens the event value was 3.2–6.9× the traced kernel at every TP. The traced `o_proj` kernel was
5.5–12 µs where the event said 36 µs. Holding the GPU busy with a spin before the timed loop (so it runs behind the host), the dip
and the plateaus vanished for every op at every TP, and the event median equalled the traced kernel plus 0.3–3.9 µs on all 32 shapes.

**Candidate 1, ruled out as the cause, but partly real.** hipBLASLt *does* switch tiles at M ≈ 4328–4840 (TP2) and 4360–4840 (TP4),
and the MT160x256 tile is 25–34 % *slower* in that band. That is a real effect, in the opposite direction of the dip, and it is still
visible as small steps in the corrected data (§5). It did not cause the dip.

**Candidate 2, ruled out.** `attn_rope` showed the same host-floor artefact acting on a 13-kernel chain. Worse (see below), the chain
was not RoPE.

**The pre-registered test (`08_`, `09_`).** Because the conclusion came from our own judgement, we wrote the predictions of both
hypotheses down before running: H0 (artefact) predicted event/kernel 3–12× at tiny shapes, H1 (real GPU effect, possibly clock-related)
predicted 0.9–1.1×. Seven falsification conditions were fixed in advance. Result: H0 stood on every decisive number; one condition, F6
(whole-forward closure), fired. The scorecard shows 262 of 369 committed numbers inside their bands and 107 outside; the misses are
reported as misses.

Things we thought that turned out wrong, kept on record:

- **Our own acceptance test was mis-specified.** T1 demanded a 1.3× cost increase from 8 to 64 tokens; the traced kernel grows only
  1.10–1.17× there because it is weight-read bound. T3 demanded a 1.5× spread across TP at 1 token; the real kernel is a fixed-cost
  `wvSplitK` at 3–6 µs. No correct instrument could pass T1/T3. They were replaced (T1′/T2) with the argument written out side by side.
- **The contamination detector we pre-registered failed its own test**: 20–34 % false positives from run-to-run drift and 2 ms jitter.
  It was not wired in; a per-row legacy/GPU-bound ratio flag is used instead.
- **F6 residual was first blamed on the instrument.** The whole-forward closure failed at 8 and 512 tokens. The mechanism turned out to
  be HIP command-queue backpressure: with 50 forwards behind one spin the queue fills, the host is throttled, and at the fastest forwards
  the device catches the host before the loop ends. 25 forwards per spin close to 0.97–0.99; 100 break everywhere.
- **Warm-up does not warm the clock.** The pre-fix profiler started its timed loop at ≈0.8 GHz on 32 of 36 grid rows *after* three
  warm-up forwards and a synchronize, ramping to ≈2.4 GHz during the loop. Three warm-up forwards do not bring an idle MI355X up from its
  idle DPM state; only continuous load does. The two timing columns therefore differ by clock state as well as by queue state.
- **The 5–11 % TP4/TP8 residual was clock, not queue.** Locking the shader clock at 1900 MHz (`rocm-smi --setperfdeterminism`, which
  works for this account) collapsed it to the +0.5…+3.6 µs instrument gap. Incident disclosed in `09_` §3b: the availability probe locked
  one GPU for two minutes outside a job before it was reset.
- **RoPE was never RoPE.** The collection script forced a torch fallback (`FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`) because a
  cookbook note said vLLM's `get_rope` signature mismatched. The fallback rotated only the first 64 columns of head 0 of the flattened
  q/k tensor, leaving 88–98 % of the elements untouched: 13 launch-bound kernels timing a computation that is not RoPE.
- **The cookbook premise was wrong for this image.** vLLM 0.9.2 in the sglang image accepts Frontier's call. But the next assumption
  was wrong too: **vLLM's own rope object on ROCm dispatches to a 17-kernel torch path** (45–160 µs), never the fused kernel. The
  fused kernel is reached by Frontier's own `RotaryEmbedding` calling `vllm._custom_ops.rotary_embedding`, and it is the same kernel
  SGLang launches on this image (timings identical to 0.1 µs). An AITER path that returned garbage turned out to be vLLM's wrapper
  around AITER, not AITER itself.
- **`sanity_check.py --data-dir <dir>` produced a vacuous PASS** twice during the work: the script takes its path positionally, so the
  option became the path, no CSV was found, and it printed PASS. Caught by the Codex review; now a usage error.

## 4. How we fixed it

Measurement, not the GPU, was the problem, so the fixes are to the profiler and to what the dataset records.

| fix | what it does |
|---|---|
| **Two timing columns** (`d4469be`) | every shape is timed twice: the old way (kept as `time_stats_hostbound.*`, documents the host-bound regime) and with the device held behind the host by a calibrated `torch.cuda._sleep` spin (`time_stats.*`, the kernel-time column). Per-row ratio and flag mark where the legacy value was a host span. |
| **Spun blocks of 25** | the GPU-bound loop re-arms the spin every 25 forwards, so any repetition count (50, 200) stays device-bound; whole-forward closure is recorded per row. |
| **Clock probe on every row** (`9dac93b`) | a fixed 2M-cycle spin timed by an event pair immediately before and after each timed loop, both passes, as an SCLK estimate; validated against a 1900 MHz lock (reads 1890–1916). A clock lock recipe with a reset trap exists for a clock-independent dataset. |
| **GC disabled around the timed loop** (`74eee7c`) | removes the 150–290 ms stall. |
| **Fused RoPE kernel** (task 1 of the recollection plan) | forced fallback removed from the script; `get_rope` prefers Frontier's class with the fused op; the fallback itself made per-head so it is never the wrong computation; every row records `attn_rope_impl`; the checker rejects anything but `vllm_kernel`. |
| **Checkers** | `sanity_check.py` requires the full two-column schema, the probe group, the rope label, coverage ≥ 3× and GPU-bound ≤ legacy where both are kernel-bound; `test_measurement_validity.py --expect red\|green` encodes the pre-registered signature. Both take the dataset path positionally. |
| **Measurement contract** (`10_`, agreed) | hetcalc's linear-op term is kernel-only device time at a stated clock condition plus a separately measured launch-overhead term, so a per-op sum never mixes host spans with kernels. |

Monotonicity was never used as a criterion for any of this. The GC fix, the two-column fix and the probe were each validated on the
9-token validation grid before the dense grid was touched. The canonical `linear_op.csv` was never modified; every new run is a new
run dir.

## 5. What the fix changed in the results

Two dense collections with every fix active (2026-09-22, node `amd-mi355x-1`): job 21483 with 25 forwards per shape (3.5 min) and job
21486 with 200 forwards as 8 spun blocks of 25 (19 min). They agree row for row (200/25 median ratio 1.000, 1st–99th percentile
0.96–1.04), so the extra repetitions bought confidence, not different numbers.

**The dip is gone.** `attn_post_proj` and `attn_rope` have no unrecovered drop anywhere in the 1k–6k band at any TP, and the plateaus
below it are gone: costs now rise with tokens as a GEMM should. What remains are small steps of 15–30 % at a handful of token counts
(4352→4360, 4480→4488, 4864→4872, 5376→5384, ≈11.8k–12.3k) at TP 2/4/8: the hipBLASLt tile switches from §3, real kernel behaviour
a cost model must represent rather than smooth. Two similar single steps appear in the norm scopes at 8192 and 14032 tokens.

**How inflated the old data was** (old median / new GPU-bound median, median over the grid):

| op | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| attn_pre_proj | 1.3× | 1.7× | 2.3× | 2.8× |
| attn_post_proj | 1.6× | 2.0× | 2.6× | 2.9× |
| attn_rope | 4.3× | 9× | 12× | 12× |

The RoPE factor combines the host-span artefact with the wrong kernel: the fused kernel is 3–47 µs across the grid where the fallback
chain was 29–100 µs.

**The stall is gone.** Over 200 samples per row, no row of any op has a sample above 10× its median (the old file had 47 such rows
for `attn_pre_proj` alone, worst 268 ms). Worst single sample anywhere: 0.74 ms.

**The instrument is accounted for.** Whole-forward closure is 0.96–0.99 on every row; the GPU-bound event exceeds the traced kernel by
a constant ≈2.5–3 µs record cost per scope boundary (12 µs for the five-kernel pre-projection scope), which is what the contract's
separate overhead term is for. The device ran at ≈2.4 GHz on all but 35 of 13,308 rows because back-to-back tasks keep it warm.

**One gate still trips, and it is the gate.** The checker's `GPU-bound ≤ 1.05 × legacy` rule was set on 32-row grids whose worst case
was 1.02. On the dense grid 875 rows (mostly TP1, 1.5k–14k tokens) read 1.05–1.09, because there the *legacy* loop sits at the
host/device crossover and its samples are bimodal (block means 36–42 µs) while the GPU-bound samples are flat (39.0 µs in all eight
blocks). The unstable column is the legacy one; the recommendation is a 1.10 bound with this reason recorded.

**Still open.** Kernels below ≈10 µs (small GEMMs at TP4/TP8, RoPE below ≈3k tokens, the norms) sit within the instrument's own floor
and are not cross-validated between event pairs and traces; the locked-clock dataset and the choice of canonical instrument
(event-pair kernel time vs kineto kernel-only) are decisions the agreement analysis is meant to inform; and the new dense runs are
still in the cluster scratch tree, not yet staged as run dirs or pointed at by the trainer.
