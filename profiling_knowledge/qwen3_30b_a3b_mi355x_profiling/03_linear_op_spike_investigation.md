# The `attn_pre_proj` run-11 spike — what it was, what we tried, what we found

Dataset: `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv` (dense grid, 3,327 token counts × TP {1,2,4,8},
collected 2026-09-15, Slurm job 21313 on `amd-mi355x-9`, image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`).

## The anomaly

- 32 rows — `num_tokens` 1968…1975 (8 consecutive values) × all four TPs — each have **exactly one** timed run of
  `attn_pre_proj` (the QKV GEMM + QK-norm) at 150–270 ms while the other 49 timed runs sit at 0.09–0.18 ms
  (~1,000–2,000× slower). It is always timed run **11** (absolute sample 14, i.e. the 15th forward including the 3 warm-ups).
- Effect on the aggregates: `median` unaffected (~0.13 ms), `mean` 3–5.5 ms, `max` 150–270 ms. This is why a mean-vs-median
  or envelope plot shows a spike at that band that does not look like ordinary jitter.
- No other op shows it. `attn_post_proj`, `attn_rope`, both layernorms and `emb` are clean in the same rows; their only
  outliers anywhere in the file are ordinary single runs of ≤ ~25× at random indices.
- The same shapes measured earlier that day on the profiler's default 386-value grid (`linear_op_grid386.csv`, tokens 1968
  and 1984 present at all TPs) show **no** spike.

## Why we tried to reproduce it

Two competing explanations needed separating: (a) a shape-triggered kernel/autotune event (hipBLASLt or aiter picking or
compiling a different GEMM kernel for M ≈ 1970, paid once per process) — deterministic, would recur; (b) a one-off stall
(external or library-internal) that happened to hit these tasks — would not recur. The choice matters for how the data is
read: (a) would be a systematic property of the shape to model or exclude, (b) is noise of a rare, large kind.

## What exactly we did

1. Read the profiler's scheduling (`frontier/profiling/linear_op/main.py`): tasks are ordered TP-outer, tokens **descending**
   inner; consecutive tasks go round-robin to the 8 GPU workers (`idx % 8`); every task builds a **fresh** `LinearOpWrapper`
   (new model, new tensors). So for each TP the eight shapes 1975…1968 started at the same instant on the 8 GPUs and ran in
   lockstep; "run 11" is therefore a fixed wall-clock offset after those eight tasks began.
2. Re-ran the band `num_tokens` 1960…1985 × TP {1,2,4,8} with the unchanged profiler (3 warm-up + 50 timed runs per shape,
   same image and node), three times — Slurm jobs 21323 and 21324 with 8 workers (same descending order and lockstep
   dispatch as the original), and 21325 with 1 worker (serial, no concurrency). Outputs under
   `/opt/shared/frontier-qwen3-profiling/Frontier/data/profiling_repro_{a,b,c}/…/linear_op.csv` on the cluster.
3. Scanned every `time_stats.<op>.samples` list in the three results (and in the original two files) for any timed run more
   than 20× its row median, recording the run index.

## Results

| Run | Setup | `attn_pre_proj` runs > 20× median | worst single run |
|---|---|---|---|
| dense sweep (job 21313) | 8 workers, 3,327 tokens | 32 rows (1968–1975 × 4 TP), all at timed index 11; plus 2 isolated rows (1992 idx 22, 2004 idx 45) | 270 ms |
| first run (job 21309, 386 tokens) | 8 workers | 0 | — |
| repro A (21323) | 8 workers, tokens 1960–1985 | 0 | 0.5 ms |
| repro B (21324) | 8 workers, same | 0 | 0.3 ms |
| repro C (21325) | 1 worker, same | 0 | 1.2 ms |

The spike is **not reproducible**: not from the shape, not from the profiler's concurrency pattern. It was a one-off event
during the dense sweep. Warm-up behaviour in the repros is unchanged (run 0 ≈ 2.1–2.5× the timed median), which also rules
out "the band needs more warm-up".

Consequences for using the data: medians (the regressor target) are untouched; 49/50 timed runs in every affected row are
normal; the rows are 32 of 13,308. Treat the event as a third noise category next to warm-up and run-to-run jitter: rare,
very large, one-off stalls inside an otherwise hot process. The fixed run index is explained by the lockstep dispatch, not by
anything the kernel does at run 11.

## Open questions and candidate causes

The lockstep explains 8 identical rows per TP. It does **not** explain why all **four** TP passes (which run ~30–40 s apart,
each sweeping the same descending token order) were stalled at the same offset within the same 8-token band. Candidates:

1. **A library-side one-time event keyed to sweep history**, e.g. hipBLASLt/Tensile lazily loading a GEMM kernel variant
   first required around M ≈ 1970 when descending from 16384, or the PyTorch caching allocator re-cutting a segment when
   activation sizes cross a rounding threshold (the QKV output crosses 20 MiB / 10 MiB at M = 2048 for TP 1 / TP 2). Against
   it: such a load would normally show on the first forward (warm-up run 0), not the 15th, and the repros with the same
   descending order did not trigger it — but the repros started at 1985, not at 16384, so the accumulated state differs.
2. **An external periodic stall on the node** (telemetry sampler, NFS flush, power/clock event) that happened to be roughly
   in phase with the ~30–40 s TP passes. Against it: landing on the same 8-token band four times is a long coincidence.
3. **Eight simultaneous first-use loads contending for the same file** (why a normally cheap event would cost ~200 ms).
   Only meaningful in combination with candidate 1.

How to discriminate, if it matters: re-run the **full** dense sweep (job 21313's command, ~2 min of GPU) once or twice. If a
similar one-run spike appears again at a different band with the lockstep signature (same index across 8 consecutive
tokens, all TPs), it is a systematic library event tied to sweep history (candidate 1) and worth a hipBLASLt log
(`HIPBLASLT_LOG_LEVEL`) or a torch profiler trace around it; if the reruns are clean, it was environmental (candidate 2).
Not done as of 2026-09-16 — the data is usable as is and the medians do not depend on the answer.
