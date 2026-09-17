# Pre-registration: validating the "host-launch-bound measurement" conclusion of `07_`

Written 2026-09-16 **before** any Phase 1–3 run. Numbers below are commitments; nothing in Phases 1–3 will be tuned,
filtered or selected on the basis of monotonicity. Where a prediction is anchored on data that already exists (the two
dense sweeps 21313/21334, the backlog sweep 21356, the rocprofv3 traces of 2026-09-16 08:38), the anchor is cited so the
reader can see that the number is a reproduction target, not a fit to the new run.

Notation: **H0** = artefact hypothesis (event pairs measure the host launch span whenever the GPU is idle when the start
event is recorded; the backlog knob only removes the idle). **H1** = the dip is a real GPU/library effect and the knob
masks it (the only mechanism found by which a GPU sleep enqueued *before* the loop could change kernel behaviour is DVFS:
a starved GPU running at lower clocks, so H1 is written out as "kernels are slower when the GPU is starved").
All times in ms unless marked µs. Tolerances are absolute unless marked %.

Methods: **(a)** current CUDA-event timing, no knob. **(b)** CUDA-event timing with `FRONTIER_GPU_BACKLOG_MS=100`.
**(c)** rocprofv3 `--kernel-trace` kernel duration (sum of the kernels inside the op's scope; the scope's kernel set is
fixed in advance: `attn_pre_proj` = Tensile GEMM + copy + `rms_norm` + copy + `rms_norm` (5 kernels), `attn_rope` = 13
kernels (`vectorized_gather`, then per q and k: mul, neg, cat, mul, add; then two `CatArrayBatchedCopy`), `attn_post_proj` =
1 Tensile GEMM). Runs: (a) and (b) through the real pipeline (`main.py`, 8 workers) **and** through the single-process
replica `trace_ops.py` (so that (c), which needs the single process, has a same-process (a)/(b) pair); the report will
say for every number which run it came from.

## 0.1 Predictions

### Phase 1 — work-independence, shapes {1, 8, 64, 4096, 8192} × TP {1,2,4,8}

**(a) current event timing.** H0 predicts a reproduction of the existing rows (job-to-job spread of those rows in
21313 vs 21334 is ≤ 10 % except in the transition region, which is why TP2@4096 gets a wide band):

| op | TP | 1 tok | 8 tok | 64 tok | 4096 tok | 8192 tok | ratio 4096/64 | ratio 64/8 |
|---|---|---|---|---|---|---|---|---|
| `attn_post_proj` | 1 | 0.019 ± 0.004 | 0.034 ± 0.005 | 0.037 ± 0.005 | 0.062 ± 0.006 | 0.094 ± 0.008 | 1.5–1.9 | 0.95–1.15 |
| `attn_post_proj` | 2 | 0.020 ± 0.004 | 0.035 ± 0.005 | 0.036 ± 0.005 | 0.037–0.062 | 0.058 ± 0.005 | 1.0–1.8 | 0.95–1.15 |
| `attn_post_proj` | 4 | 0.021 ± 0.004 | 0.035 ± 0.005 | 0.036 ± 0.005 | 0.050 ± 0.006 | 0.056 ± 0.006 | 1.2–1.6 | 0.95–1.15 |
| `attn_post_proj` | 8 | 0.020 ± 0.004 | 0.036 ± 0.006 | 0.035 ± 0.005 | 0.046 ± 0.005 | 0.052 ± 0.005 | 1.1–1.5 | 0.95–1.15 |
| `attn_rope` | 1 | 0.072 ± 0.010 | 0.082 ± 0.010 | 0.084 ± 0.010 | 0.078 ± 0.008 | 0.100 ± 0.008 | 0.8–1.1 | 0.9–1.15 |
| `attn_rope` | 2 | 0.078 ± 0.012 | 0.088 ± 0.015 | 0.091 ± 0.015 | 0.064 ± 0.008 | 0.083 ± 0.006 | 0.6–0.85 | 0.9–1.15 |
| `attn_rope` | 4 | 0.080 ± 0.012 | 0.088 ± 0.012 | 0.092 ± 0.012 | 0.092 ± 0.010 | 0.067 ± 0.006 | 0.85–1.15 | 0.9–1.15 |
| `attn_rope` | 8 | 0.080 ± 0.012 | 0.090 ± 0.015 | 0.087 ± 0.012 | 0.093 ± 0.008 | 0.080 ± 0.012 | 0.9–1.2 | 0.9–1.15 |
| `attn_pre_proj` | 1 | 0.054 ± 0.008 | 0.083 ± 0.010 | 0.084 ± 0.010 | 0.182 ± 0.012 | 0.337 ± 0.015 | 2.0–2.4 | 0.9–1.15 |
| `attn_pre_proj` | 2 | 0.061 ± 0.008 | 0.092 ± 0.012 | 0.092 ± 0.012 | 0.127 ± 0.012 | 0.184 ± 0.010 | 1.2–1.55 | 0.9–1.15 |
| `attn_pre_proj` | 4 | 0.061 ± 0.008 | 0.093 ± 0.012 | 0.093 ± 0.012 | 0.092 ± 0.010 | 0.107 ± 0.010 | 0.85–1.15 | 0.9–1.15 |
| `attn_pre_proj` | 8 | 0.061 ± 0.008 | 0.095 ± 0.012 | 0.092 ± 0.012 | 0.084 ± 0.010 | 0.070 ± 0.012 | 0.8–1.05 | 0.9–1.15 |

The claim under H0 is the two right-hand columns: an 8× change in work (8 → 64 tokens) moves the measured value by
< 15 % for every op and TP, and a 64× change (64 → 4096) moves `attn_post_proj` by < 60 % at TP 4/8 while the kernel (see (c))
grows ≥ 3×. H0 also predicts the 1-token row sits on a *different, lower* host floor than 8/64 tokens (0.020 vs 0.035 for
`attn_post_proj`; anchored on 21313/21334). I cannot commit to why (§0.3, item 1); H0 only needs that it is TP-independent
(±20 % across TP 1→8 while K changes 8×). **H1** (event values are GPU times) has no reason for the 1-token value to be
TP-independent within 20 % while the weight read changes 8×; it also predicts (c) ≈ (a) within 10 % everywhere.

**(b) event timing with the backlog knob.** H0: value = kernel-sum + one ≈6 µs dispatch gap per kernel for GEMMs, and
≈ the dispatch floor (13 × 4.6–4.8 µs) for the RoPE chain (anchored: X1 rows 4096/8192; traced backlog run: rope kernels
back-to-back at 4.6–4.8 µs each with 0.0 idle).

| op | TP | 1 tok | 8 tok | 64 tok | 4096 tok | 8192 tok | ratio 4096/1 |
|---|---|---|---|---|---|---|---|
| `attn_post_proj` | 1 | 0.008–0.016 | 0.008–0.016 | 0.009–0.018 | 0.063 ± 0.004 | 0.097 ± 0.005 | ≥ 3.5 |
| `attn_post_proj` | 2 | 0.007–0.014 | 0.007–0.014 | 0.008–0.016 | 0.037 ± 0.003 | 0.059 ± 0.003 | ≥ 2.5 |
| `attn_post_proj` | 4 | 0.007–0.013 | 0.007–0.013 | 0.008–0.015 | 0.024 ± 0.003 | 0.037 ± 0.003 | ≥ 1.7 |
| `attn_post_proj` | 8 | 0.007–0.013 | 0.007–0.013 | 0.007–0.014 | 0.019 ± 0.003 | 0.026 ± 0.003 | ≥ 1.3 |
| `attn_rope` | any | 0.050–0.068 | 0.050–0.068 | 0.050–0.068 | TP1 0.080 ± 0.005, TP2 0.064 ± 0.004, TP4 0.061 ± 0.004, TP8 0.057 ± 0.004 | TP1 0.100, TP2 0.083, TP4 0.070, TP8 0.062 (± 0.005) | **0.9–1.35** |
| `attn_pre_proj` | 1 | 0.024–0.036 | 0.024–0.036 | 0.026–0.040 | 0.184 ± 0.008 | 0.340 ± 0.012 | ≥ 5 |
| `attn_pre_proj` | 2 | 0.022–0.034 | 0.022–0.034 | 0.024–0.038 | 0.099 ± 0.006 | 0.185 ± 0.008 | ≥ 2.8 |
| `attn_pre_proj` | 4 | 0.022–0.034 | 0.022–0.034 | 0.023–0.036 | 0.063 ± 0.005 | 0.101 ± 0.006 | ≥ 1.8 |
| `attn_pre_proj` | 8 | 0.022–0.034 | 0.022–0.034 | 0.023–0.036 | 0.047 ± 0.004 | 0.066 ± 0.005 | ≥ 1.3 |

Declared in advance: **for `attn_rope`, method (b) will also look work-independent at small shapes** (ratio 0.9–1.35), and
so will (c) partially, because 13 back-to-back kernels of 2–5 µs are bounded by per-kernel fixed cost, not by data volume.
That is a property of the op as implemented (the torch fallback), not a failure of the method; it is why the
work-independence test is decisive for the GEMMs and only weakly informative for `attn_rope`. **H1** predicts (b) ≈ (a)
within 10 % (the knob would not change a real GPU cost) except where it claims DVFS; under the DVFS form of H1 (b) < (a)
but then (c) measured *without* the knob must also be ≥ 15 % longer than (c) with the knob at the same shape (§ Phase 2, D).

**(c) rocprofv3 kernel durations (µs), no knob.** Anchors: traces of 08:38 (TP2 3000 → 28.6, 4128 → 35.8, 8000 → 56.0;
TP4 3000 → 18.7, 4128 → 23.3, 8000 → 33.6; QKV TP2 3000 → 32.7, 8000 → 92; rope chain TP2 3000 → ≈58, 8000 → ≈90).

| op | TP | 1 tok | 8 tok | 64 tok | 4096 tok | 8192 tok | ratio 4096/1 |
|---|---|---|---|---|---|---|---|
| `attn_post_proj` GEMM | 1 | 3–10 | 3–10 | 4–12 | 55–64 | 88–96 | ≥ 5 |
| `attn_post_proj` GEMM | 2 | 3–9 | 3–9 | 3–10 | 33–38 | 54–60 | ≥ 3.5 |
| `attn_post_proj` GEMM | 4 | 2–8 | 2–8 | 3–9 | 20–26 | 32–37 | ≥ 2.5 |
| `attn_post_proj` GEMM | 8 | 2–8 | 2–8 | 3–9 | 11–17 | 18–24 | ≥ 1.4 |
| `attn_rope` 13-kernel sum | any | 25–65 | 25–65 | 25–65 | TP1 70–90, TP2 55–70, TP4 50–65, TP8 45–62 | TP1 90–110, TP2 80–95, TP4 60–75, TP8 55–68 | 0.9–2.5 |
| `attn_pre_proj` 5-kernel sum | 1 | 15–35 | 15–35 | 16–38 | 170–190 | 320–350 | ≥ 4.5 |
| `attn_pre_proj` 5-kernel sum | 2 | 14–32 | 14–32 | 15–35 | 88–100 | 175–190 | ≥ 2.7 |
| `attn_pre_proj` 5-kernel sum | 4 | 14–32 | 14–32 | 15–35 | 55–65 | 92–104 | ≥ 1.7 |
| `attn_pre_proj` 5-kernel sum | 8 | 14–32 | 14–32 | 15–35 | 40–48 | 58–66 | ≥ 1.25 |

**(a) vs (c) at tiny shapes is the sharpest H0/H1 split:** H0 predicts (a)/(c) for `attn_post_proj` at 8–64 tokens is
3–12× at every TP; H1 predicts 0.9–1.1×.

**Plateau vs K from existing data** (already computable, restated as a prediction for the new (a) rows at 3072 tokens):
(a) `attn_post_proj` at 2800–3200 tokens is 0.051 (TP1, K=4096), 0.058 (TP2, 2048), 0.049 (TP4, 1024), 0.047 (TP8, 512): an
8× change in K moves the measured value by −8 % and non-monotonically. (b)/(c) for the same rows: 0.052 / 0.033 / 0.022 /
0.0215 (X1) → kernel ≈ 46 / 28.6 / 18.7 / 12–16 µs; the TP4→TP8 step is predicted to shrink the kernel by only 15–30 % (the
GEMM is approaching its fixed-cost floor), which is a real non-monotonic-in-efficiency effect and will be kept as is.

### Phase 2 — closure, TP2 and TP4 at {3072, 4192, 6144, 8192} (plus TP1/TP8 for the same tokens as a bonus)

Definitions fixed now. **W** = `perf_counter` immediately before the timed loop to immediately after the trailing
`torch.cuda.synchronize()`, divided by 50 (a new field `host_wall_per_forward_ms`; Phase 3 will audit that it touches no
timer arithmetic). In the backlog run the same field measures (100 ms sleep + 50 forwards)/50, so **W′ = W − 100/50** is
reported for that run and must not be compared to W. **G_sum** = sum of all kernel durations of one forward (all 41 kernels,
MLP and `randn_like` included, since they run); **G_span** = first kernel start to last kernel end of the forward; **G_op** =
kernel sum inside each op's scope as fixed above. **E** = (a) medians; **B** = (b) medians; sums over the three attention
ops are written E₃, B₃, G₃. W exists in the untraced (a) run (W_a), the traced run (W_c; inflated by the tracer), and W′ in
the (b) run. The tracer adds per-launch host cost, so W_c ≠ W_a is expected, not a failure; both are reported.

Predicted values under H0 (TP2; anchors: E1b host time 0.54–0.57 ms/forward at 8 workers; traced G at 3000/4128/6432/8000
= 316/375/520/619 µs, TP4 = 274/311/384/448 µs; single-process host span may be up to 20 % shorter than the 8-worker one):

| TP | tokens | W_a | W_c | G_sum | G₃ | E₃ | B₃ | idle = 1 − G_sum/W_a |
|---|---|---|---|---|---|---|---|---|
| 2 | 3072 | 0.44–0.60 | 0.65–0.85 | 0.30–0.36 | 0.15–0.18 | 0.22–0.27 | 0.16–0.19 | 25–45 % |
| 2 | 4192 | 0.44–0.60 | 0.65–0.85 | 0.35–0.41 | 0.17–0.20 | 0.21–0.27 | 0.18–0.21 | 15–40 % |
| 2 | 6144 | 0.44–0.60 | 0.65–0.85 | 0.47–0.53 | 0.24–0.28 | 0.24–0.30 | 0.25–0.29 | 0–20 % |
| 2 | 8192 | max(0.44–0.60, G_sum) ± 10 % | 0.70–0.90 | 0.60–0.68 | 0.31–0.35 | 0.31–0.36 | 0.32–0.36 | 0–8 % |
| 4 | 3072 | 0.44–0.60 | 0.65–0.85 | 0.26–0.31 | 0.11–0.13 | 0.20–0.26 | 0.12–0.14 | 35–55 % |
| 4 | 4192 | 0.44–0.60 | 0.65–0.85 | 0.29–0.34 | 0.12–0.14 | 0.20–0.26 | 0.13–0.15 | 30–50 % |
| 4 | 6144 | 0.44–0.60 | 0.65–0.85 | 0.36–0.41 | 0.14–0.17 | 0.20–0.26 | 0.15–0.18 | 20–40 % |
| 4 | 8192 | 0.44–0.60 | 0.65–0.85 | 0.43–0.48 | 0.17–0.20 | 0.21–0.27 | 0.18–0.21 | 10–30 % |

Identities, pass/fail per row, committed now:

- **A. B₃ ≤ G_sum ≤ W_a**: pass on every row with margin ≥ 30 % on the first inequality except TP2@8192, where G_sum ≈ W_a
  (the GPU becomes the bottleneck) and the second inequality is expected to hold only within ±10 %.
- **A′. Per-op B_op vs G_op**: **predicted to FAIL the strict form** by a fixed amount: B_op − G_op = +4 to +12 µs for the
  GEMM ops (one ≈6 µs dispatch gap per kernel boundary inside the scope) and −5 to +10 µs for `attn_rope` (its kernels run
  back-to-back, so the gaps are already inside the traced durations). This is the backlog method's known bias and will be
  reported as a failure of the strict inequality, not reconciled.
- **B. E vs W and G**: E_op/G_op at TP2@3072 = 1.3–1.7 (pre), 1.2–1.5 (rope), 1.7–2.3 (post); at TP2@8192 all three 0.95–1.08;
  at TP4@8192 post 1.5–1.9, rope 0.95–1.1, pre 1.0–1.15. E₃ < W_a on every row (the three scopes cover only 19 of 41 kernels
  and none of the untimed host work), so "E tracks W" is tested as: E_op is flat in tokens (±15 %) while G_op grows ≥ 1.8×
  between 3072 and 8192 for `attn_post_proj` at TP4 (a host span does not know M), and E_op collapses onto G_op exactly where
  the idle gap before the op's first kernel (from the trace) falls below ≈10 µs.
- **C. Idle fraction**: the `07_` figure "45–60 % idle at 3000–8000 tokens" came from traced runs (W_c ≈ 0.75). With W_a I
  predict lower values, 25–45 % at TP2@3072 falling to 0–8 % at 8192, and 35–55 % → 10–30 % at TP4. **If the untraced idle
  at TP2@3072 is < 20 %, the starvation claim as stated in `07_` is wrong in magnitude and I will say so.**
- **D. DVFS / masking check (decisive for H1)**: kernel durations of the same shape traced with and without the knob agree
  within 5 % for all GEMMs and within 10 % for the sub-10 µs kernels (anchor: 28.6 vs 28.5, 56.0 vs 56.6 µs). H1-DVFS predicts
  the no-knob kernels are ≥ 15 % longer. Bonus: `rocm-smi --showclocks` sampled during both runs; H0 predicts the same
  sclk state, H1-DVFS a lower one in the no-knob run.
- **E. Backlog closure**: W′ (backlog run) = G_sum ± 10 % on every row (after the sleep the GPU runs 50 forwards
  back-to-back, so wall/50 must equal GPU busy per forward plus dispatch gaps). This is a test of the knob itself that does
  not use any event pair.

### Phase 3 — self-audit

Predictions are trivial but stated: the diff touches (i) `linear_op_wrapper.py`: constants, one helper, one `if` before the
timed loop, and (for Phase 2) one `perf_counter` pair and one new dict entry; (ii) the sbatch: two `${VAR:-}` passthroughs;
(iii) `trace_ops.py`: new file; (iv) changes by the other session (`ACTIVE_STEPS` env override, `spike_diag` hooks, `main.py`
hooks) which I will list but did not write. None touches `cuda_timer.py`, `timer_stats_store.py`, event pairing, or the
median/std; the median is over the same 50 timed runs. Any hunk that contradicts this is a finding.

### Phase 4 — adversarial case

Committed in advance: the strongest H1 case is DVFS (starved GPU at lower clock) plus the real tile-swap slow-downs. What
would defeat it is check D (equal kernel durations with and without the knob) together with Phase 1 (a)-vs-(c) at 8–64
tokens (a GEMM reading 2–16 MB of weights cannot take 35 µs on a 8 TB/s device unless the clock is < 1/10 of nominal, which
`rocm-smi` would show). If D fails (≥ 15 % longer kernels without the knob), I concede that part of the "dip" is GPU
behaviour and the knob changes it.

### Data handling — contamination detector, fixed now

Per row and op: `host_bound_suspect = (row_std > 5 × baseline_op) OR (|mean(runs 0–9) − mean(runs 40–49)| / median > 10 %)`
with **fixed** baselines (median row std of rows where 21313 agrees with 21356 within 5 %): `attn_post_proj` 0.0013,
`attn_rope` 0.0015, `attn_pre_proj` 0.0075 ms. Predicted performance, stated before it is run on the full files:
- rows where 21313/21356 ≥ 1.2 (known contaminated): flagged fraction 60–85 %, **not higher** — rows with a fast, steady
  host sit on a tight floor (e.g. TP2@3000: p10–p90 0.053–0.057) and will be missed; the 1-token rows (std 0.002, drift 0 %)
  will all be missed although they are 2–4× GPU time under H0. The detector is therefore necessary-not-sufficient, and the
  report will say so rather than tune it.
- rows where 21313/21356 within 5 %: flagged ≤ 10 %. Backlog-run rows: flagged ≤ 5 %.
It is implemented as a column written by the pipeline plus a `sanity_check.py` assertion that fails unless
`--allow-host-bound` is passed; canonical CSVs are not modified (a flagged copy is written next to them).

### RoPE fallback numeric check (independent of timing)

Prediction from reading `rotary_embedding.py`: for `q` of shape `[N, heads·128]`, the fallback leaves columns 64…end
bit-identical to the input (fraction of untouched elements = 1 − 64/(heads·128): 0.984 at TP1, 0.969 at TP2, 0.938 at TP4,
0.875 at TP8) and rotates columns 0–63 with `_rotate_half` over a 64-wide slice, i.e. pairs (i, i+32) with `cos/sin` of the
first 64 frequencies of the cache. Against a reference neox RoPE over 128-dim heads (pairs (i, i+64) per head), max abs
difference is O(1) for random inputs at every position except position 0. If the untouched fraction is not within 0.005
of the numbers above, the code reading in `07_` §3.5 is wrong.

## 0.2 Falsification conditions (any one → I retract the artefact conclusion, or the part named)

1. Phase 1 (c): `attn_post_proj` GEMM kernel at 8 or 64 tokens ≥ 20 µs at any TP, **or** (c) ratio 4096/64 < 2.0 at TP 1/2/4
   → the GEMM really is work-independent on the GPU; H0's central claim fails.
2. Phase 1 (a) vs (c): (a)/(c) for `attn_post_proj` at 8–64 tokens < 1.5 at TP2 or TP4 → the event value is the kernel; H0 fails.
3. Phase 2 D: no-knob kernel durations ≥ 15 % longer than with-knob durations at the same shape for the `o_proj` GEMM →
   the knob changes GPU behaviour; the "identical GPU work" claim is retracted and H1-DVFS is live.
4. Phase 2 A: B₃ > G_sum on any row → the knob adds time inside the timed scopes; the knob is retracted as a measurement.
5. Phase 2 C: untraced idle at TP2@3072 < 20 % **and** at TP4@3072 < 25 % → the starvation magnitude in `07_` is wrong; the
   mechanism could still hold locally (idle before the op's first kernel) but the report's framing is retracted.
6. Phase 2 E: W′ deviates from G_sum by > 15 % on more than one row → the backlog run is not the clean GPU-bound loop I
   claimed; (b) numbers are downgraded to "unexplained".
7. Phase 1 (a): the 1-token `attn_post_proj` value varies by > 40 % across TP 1→8 in the same run → the "TP-independent host
   floor" reading of that row is wrong (this would not by itself rescue H1, but it would show H0 does not explain the
   small-shape rows and that would be reported as such).

## 0.3 Predictions I cannot commit to

1. **Why the 1-token rows sit on a lower host floor** (0.020 vs 0.035 ms for `attn_post_proj`, 0.055 vs 0.09 for
   `attn_pre_proj`) than 8/64 tokens. Candidates are a different hipBLASLt/rocBLAS path for M = 1 with a shorter host path,
   or a different torch dispatch path; I have not read that code and will not guess a number. H0 requires only
   TP-independence of the row (§0.2 item 7).
2. **Exact sub-10 µs kernel durations** at 1–64 tokens (ranges only, above). The traced durations of tiny kernels were
   2.2–5.0 µs when the GPU was idle between them and 4.6–4.8 µs when queued back-to-back; whether that is a rocprofv3
   timestamp granularity effect or real dispatch behaviour I cannot say in advance, so (c) for `attn_rope` is given as a range.
3. **W_a in a single process vs. 8 workers**: I expect 0.44–0.60 ms and cannot narrow it; the 8-worker E1b figure (0.54–0.57)
   may include contention that a single process lacks.
4. **Whether `rocm-smi` clock sampling will be fine-grained enough** to see a DVFS state during a 2–3 s task. If it is not,
   check D rests on kernel durations alone, which I consider sufficient but say so here.
5. **The detector's exact sensitivity** (60–85 % is a band, not a point).

## What will be run after approval

1. Pipeline jobs (8 workers, sbatch, new `COLLECT_DIR`s, canonical CSV untouched): (a) and (b) on
   `{1, 8, 64, 3072, 4096, 4192, 6144, 8192}` × TP {1,2,4,8}; both with the new `host_wall_per_forward_ms` field.
2. Single-process `trace_ops.py` on the same shapes: (a) untraced, (b) untraced with knob, (c) rocprofv3 without knob,
   (c′) rocprofv3 with knob; `rocm-smi --showclocks` sampled every 0.5 s alongside (c) and (c′).
3. Detector column + `sanity_check.py` assertion; run on 21313, 21334, 21356 and the new files; report flagged fractions
   against the §0.1 predictions.
4. RoPE fallback numeric check on CPU against a reference implementation.
5. Full diff and Phase 3 answers; Phase 4 written after the numbers are in, using them.
