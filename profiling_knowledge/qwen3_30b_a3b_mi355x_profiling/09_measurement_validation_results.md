# Measurement validation results (answers `08_measurement_validation_preregistration.md`)

Written 2026-09-17. Data: jobs 21376 (single-process traces: event timing, event + backlog, rocprofv3, rocprofv3 + backlog; tokens
{1, 8, 64, 3072, 4096, 4192, 6144, 8192} × TP {1,2,4,8}, GPU 0 of `amd-mi355x-7`), 21377/21378 (8-worker pipeline, same grid, legacy / knob),
21389 (fixed two-pass profiler, same grid → `runs/2026-09-17_0846_linear_op_validation_grid_two_column/`). Full debug-loop record:
`amd-playground/.claude/debug-reports/debug_linear_op_host_bound_timing_2026-09-16.md` (root cause, Stage 4) and
`implement_linear_op_two_column_timing_2026-09-17.md` (fix). Ground rule kept throughout: monotonicity was not used as evidence anywhere.

## 1. Verdict on the hypotheses

**H0 (artefact) stands; H1 (real GPU/library effect masked by the knob) is refuted on its own predictions.** The decisive numbers:
`attn_post_proj` event/kernel at 8–64 tokens = 3.2–6.9 at every TP (H0 predicted 3–12, H1 predicted 0.9–1.1); the traced o_proj kernel
is 5.5–12.1 µs at 8–64 tokens and 28–36 µs at 3072–4192 (H1 needed ≥ 35 µs at 8–64 tokens); the no-knob/knob kernel ratio for the o_proj
and QKV GEMMs at ≥ 3072 tokens is 0.98–1.11 (H1-DVFS needed ≥ 1.15, a 3–7× gap needed ≥ 3); with the knob the event median equals the
traced kernel + 0.3…+3.9 µs on all 32 shapes and is never below it. One falsification condition fired (F6, §3); it concerns the
whole-forward accounting of the backlog method, not the per-op claim.

## 2. Acceptance test: original vs. revised, side by side

| | Original (Stage 1 success criterion, written before the runs) | Revised (T1′/T2, user-accepted 2026-09-17) |
|---|---|---|
| T1 | `attn_post_proj` median(64 tok) / median(8 tok) ≥ 1.30 at every TP | **T1′**: legacy/GPU-bound ratio ≥ 2.0 at 8 and 64 tokens at every TP, **and** GPU-bound ≤ 1.05 × legacy on every row |
| T2 | median(4096) / median(64) ≥ 2.0 at TP 1/2/4 | unchanged, on the GPU-bound column |
| T3 | 1-token median max/min across TP ≥ 1.5 | dropped for the GPU-bound column; kept on the legacy column as the RED signature |
| result on legacy data (21313/21334/21377) | RED 8/8 (T1 0.991–1.108, T2 1.34–1.74, T3 1.06–1.11) | legacy columns: RED as required |
| result on the knob run (21378) | **RED 3/8**: T1 TP1 1.129, TP2 1.108; T3 1.243; T2 4.77/3.11/2.11 pass | — |
| result on the fixed profiler (21389) | — | GREEN: T1′ 2.95–5.33, max GPU-bound/legacy 1.005; T2 4.56/2.97/2.02 |

**Argument for the revision (post-hoc, from the kernel traces, not from the desired outcome):** T1 assumed compute-bound scaling, but the
traced o_proj kernel grows only 1.10–1.17× from 8 to 64 tokens at TP1/TP2 (weight-read bound: 16/8 MB of weights dominate at tiny M) and
1.64–1.86× at TP4/TP8 — so no correct instrument can pass T1 at TP1/TP2. T3 assumed the 1-token GEMM cost scales with K; the traced
1-token kernel is 3.0–6.2 µs with a spread of 1.8× across TP (fixed-cost bound) and takes a different kernel (`wvSplitK_hf_sml_`), and one
≈3 µs dispatch gap compresses the event spread to 1.24. T1′ tests the quantity the fix is about (does the legacy column exceed the
GPU-bound one where the GPU was idle) and T2 keeps the work-dependence test where the kernel genuinely scales (traced 4.98/3.37/2.15).
The T2 TP4 margin is 3–7 % and is a known flake risk to be reported, not re-thresholded. **Weakness kept on record:** the "legacy must fail"
half of `--expect green` is satisfied by T1's mis-specification alone, so it is a weak check; the decisive half is T1′/T2.

## 3. F6 and the other failed conditions
See the debug report's "failed first" section for the full list. F6 (whole-forward closure `W′ ≈ Σ kernel durations`) fired: W′ exceeds the
kernel sum by 8–53 %. The traced gap structure locates the excess at the profiler's own 20 event records per forward (30 of 41 kernel
boundaries have 0.00 µs gap; 11 carry all of it at event positions), plus an unexplained ≈40 µs at 8 tokens. The per-op claim does not use
this identity. The fixed profiler now records `time_stats.forward_gpu_span` so closure is measured on every row: 0.985–0.991 at ≥ 3072
tokens, 0.965/0.971 at 1/64 tokens, **0.74–0.79 at 8 tokens** on run 21389 (see RUN.md); the 8-token residual is open.

## 4. Cross-instrument agreement — what is and is not covered
Event (GPU-bound) vs traced kernel agree within +0.3…+3.9 µs. That is 1–7 % for kernels ≥ 20 µs (o_proj at TP1 ≥ 8 tokens, TP2 ≥ 3072,
QKV at every TP ≥ 3072) and **20–80 % for kernels ≤ 10 µs** (o_proj at TP2/4/8 below ~1000 tokens, the 1-token path everywhere, the
individual RoPE/norm kernels). Rows in the dataset whose GPU-bound `attn_post_proj` median is below ≈ 0.020 ms (TP4/TP8 below ~3000 tokens,
TP2 below ~1500, TP1 below ~500) should be read as *un-cross-validated*: the instrument's own ≈3 µs floor is a large fraction of the value.
`attn_rope` is un-cross-validatable in either column (13-kernel torch fallback, numerically wrong).

## 5. Scorecard — every committed number
Generated 2026-09-17 by `scorecard.py` from the raw outputs of jobs 21376 (single-process traces, modes a/b/c/c′), and the detector/RoPE
checks below. **262 of 369 committed numbers inside their band; 107 outside.** Bands are copied verbatim from 08_; observed values are
medians over the 50 timed runs (event methods) or over the 49 fully-bracketed timed forwards (trace). Nothing was re-tuned after the fact:
misses are shown as misses. Falsification conditions (08_ §0.2): F1 no, F2 no, F3 no (o_proj/QKV ≤ 11 %), F4 no, F5 no, **F6 fired**, F7 no.

| section | item | pre-registered band | observed | result |
|---|---|---|---|---|
| Phase 1 (a) event timing | attn_post_proj TP1 @1 tok (ms) | 0.0150–0.0230 | 0.0213 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP1 @8 tok (ms) | 0.0290–0.0390 | 0.0357 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP1 @64 tok (ms) | 0.0320–0.0420 | 0.0381 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP1 @4096 tok (ms) | 0.0560–0.0680 | 0.0631 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP1 @8192 tok (ms) | 0.0860–0.1020 | 0.0966 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP1 ratio 4096/64 | 1.50–1.90 | 1.66 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP1 ratio 64/8 | 0.95–1.15 | 1.07 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP2 @1 tok (ms) | 0.0160–0.0240 | 0.0215 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP2 @8 tok (ms) | 0.0300–0.0400 | 0.0347 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP2 @64 tok (ms) | 0.0310–0.0410 | 0.0366 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP2 @4096 tok (ms) | 0.0370–0.0620 | 0.0615 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP2 @8192 tok (ms) | 0.0530–0.0630 | 0.0592 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP2 ratio 4096/64 | 1.00–1.80 | 1.68 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP2 ratio 64/8 | 0.95–1.15 | 1.06 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP4 @1 tok (ms) | 0.0170–0.0250 | 0.0213 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP4 @8 tok (ms) | 0.0300–0.0400 | 0.0365 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP4 @64 tok (ms) | 0.0310–0.0410 | 0.0378 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP4 @4096 tok (ms) | 0.0440–0.0560 | 0.0510 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP4 @8192 tok (ms) | 0.0500–0.0620 | 0.0649 | **FAIL** |
| Phase 1 (a) event timing | attn_post_proj TP4 ratio 4096/64 | 1.20–1.60 | 1.35 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP4 ratio 64/8 | 0.95–1.15 | 1.04 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP8 @1 tok (ms) | 0.0160–0.0240 | 0.0217 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP8 @8 tok (ms) | 0.0300–0.0420 | 0.0380 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP8 @64 tok (ms) | 0.0300–0.0400 | 0.0386 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP8 @4096 tok (ms) | 0.0410–0.0510 | 0.0474 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP8 @8192 tok (ms) | 0.0470–0.0570 | 0.0541 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP8 ratio 4096/64 | 1.10–1.50 | 1.23 | PASS |
| Phase 1 (a) event timing | attn_post_proj TP8 ratio 64/8 | 0.95–1.15 | 1.01 | PASS |
| Phase 1 (a) event timing | attn_rope TP1 @1 tok (ms) | 0.0620–0.0820 | 0.0872 | **FAIL** |
| Phase 1 (a) event timing | attn_rope TP1 @8 tok (ms) | 0.0720–0.0920 | 0.0943 | **FAIL** |
| Phase 1 (a) event timing | attn_rope TP1 @64 tok (ms) | 0.0740–0.0940 | 0.1003 | **FAIL** |
| Phase 1 (a) event timing | attn_rope TP1 @4096 tok (ms) | 0.0700–0.0860 | 0.0791 | PASS |
| Phase 1 (a) event timing | attn_rope TP1 @8192 tok (ms) | 0.0920–0.1080 | 0.0996 | PASS |
| Phase 1 (a) event timing | attn_rope TP1 ratio 4096/64 | 0.80–1.10 | 0.79 | **FAIL** |
| Phase 1 (a) event timing | attn_rope TP1 ratio 64/8 | 0.90–1.15 | 1.06 | PASS |
| Phase 1 (a) event timing | attn_rope TP2 @1 tok (ms) | 0.0660–0.0900 | 0.0916 | **FAIL** |
| Phase 1 (a) event timing | attn_rope TP2 @8 tok (ms) | 0.0730–0.1030 | 0.0991 | PASS |
| Phase 1 (a) event timing | attn_rope TP2 @64 tok (ms) | 0.0760–0.1060 | 0.1018 | PASS |
| Phase 1 (a) event timing | attn_rope TP2 @4096 tok (ms) | 0.0560–0.0720 | 0.0673 | PASS |
| Phase 1 (a) event timing | attn_rope TP2 @8192 tok (ms) | 0.0770–0.0890 | 0.0826 | PASS |
| Phase 1 (a) event timing | attn_rope TP2 ratio 4096/64 | 0.60–0.85 | 0.66 | PASS |
| Phase 1 (a) event timing | attn_rope TP2 ratio 64/8 | 0.90–1.15 | 1.03 | PASS |
| Phase 1 (a) event timing | attn_rope TP4 @1 tok (ms) | 0.0680–0.0920 | 0.0858 | PASS |
| Phase 1 (a) event timing | attn_rope TP4 @8 tok (ms) | 0.0760–0.1000 | 0.0950 | PASS |
| Phase 1 (a) event timing | attn_rope TP4 @64 tok (ms) | 0.0800–0.1040 | 0.0978 | PASS |
| Phase 1 (a) event timing | attn_rope TP4 @4096 tok (ms) | 0.0820–0.1020 | 0.0973 | PASS |
| Phase 1 (a) event timing | attn_rope TP4 @8192 tok (ms) | 0.0610–0.0730 | 0.0666 | PASS |
| Phase 1 (a) event timing | attn_rope TP4 ratio 4096/64 | 0.85–1.15 | 0.99 | PASS |
| Phase 1 (a) event timing | attn_rope TP4 ratio 64/8 | 0.90–1.15 | 1.03 | PASS |
| Phase 1 (a) event timing | attn_rope TP8 @1 tok (ms) | 0.0680–0.0920 | 0.0872 | PASS |
| Phase 1 (a) event timing | attn_rope TP8 @8 tok (ms) | 0.0750–0.1050 | 0.0977 | PASS |
| Phase 1 (a) event timing | attn_rope TP8 @64 tok (ms) | 0.0750–0.0990 | 0.0989 | PASS |
| Phase 1 (a) event timing | attn_rope TP8 @4096 tok (ms) | 0.0850–0.1010 | 0.0994 | PASS |
| Phase 1 (a) event timing | attn_rope TP8 @8192 tok (ms) | 0.0680–0.0920 | 0.0951 | **FAIL** |
| Phase 1 (a) event timing | attn_rope TP8 ratio 4096/64 | 0.90–1.20 | 1.00 | PASS |
| Phase 1 (a) event timing | attn_rope TP8 ratio 64/8 | 0.90–1.15 | 1.01 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP1 @1 tok (ms) | 0.0460–0.0620 | 0.0594 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP1 @8 tok (ms) | 0.0730–0.0930 | 0.0880 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP1 @64 tok (ms) | 0.0740–0.0940 | 0.0913 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP1 @4096 tok (ms) | 0.1700–0.1940 | 0.1835 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP1 @8192 tok (ms) | 0.3220–0.3520 | 0.3328 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP1 ratio 4096/64 | 2.00–2.40 | 2.01 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP1 ratio 64/8 | 0.90–1.15 | 1.04 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP2 @1 tok (ms) | 0.0530–0.0690 | 0.0625 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP2 @8 tok (ms) | 0.0800–0.1040 | 0.0951 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP2 @64 tok (ms) | 0.0800–0.1040 | 0.0953 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP2 @4096 tok (ms) | 0.1150–0.1390 | 0.1284 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP2 @8192 tok (ms) | 0.1740–0.1940 | 0.1849 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP2 ratio 4096/64 | 1.20–1.55 | 1.35 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP2 ratio 64/8 | 0.90–1.15 | 1.00 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP4 @1 tok (ms) | 0.0530–0.0690 | 0.0618 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP4 @8 tok (ms) | 0.0810–0.1050 | 0.0944 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP4 @64 tok (ms) | 0.0810–0.1050 | 0.0958 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP4 @4096 tok (ms) | 0.0820–0.1020 | 0.0910 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP4 @8192 tok (ms) | 0.0970–0.1170 | 0.1126 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP4 ratio 4096/64 | 0.85–1.15 | 0.95 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP4 ratio 64/8 | 0.90–1.15 | 1.01 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP8 @1 tok (ms) | 0.0530–0.0690 | 0.0631 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP8 @8 tok (ms) | 0.0830–0.1070 | 0.0973 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP8 @64 tok (ms) | 0.0800–0.1040 | 0.0976 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP8 @4096 tok (ms) | 0.0740–0.0940 | 0.0874 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP8 @8192 tok (ms) | 0.0580–0.0820 | 0.0777 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP8 ratio 4096/64 | 0.80–1.05 | 0.90 | PASS |
| Phase 1 (a) event timing | attn_pre_proj TP8 ratio 64/8 | 0.90–1.15 | 1.00 | PASS |
| Phase 1 (a) event timing | attn_post_proj 1-token max/min across TP (TP-independent floor) | 1.00–1.20 | 1.02 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP1 @1 tok (ms) | 0.0080–0.0160 | 0.0075 | **FAIL** |
| Phase 1 (b) event + backlog | attn_post_proj TP1 @8 tok (ms) | 0.0080–0.0160 | 0.0120 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP1 @64 tok (ms) | 0.0090–0.0180 | 0.0134 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP1 @4096 tok (ms) | 0.0590–0.0670 | 0.0637 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP1 @8192 tok (ms) | 0.0920–0.1020 | 0.0969 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP1 ratio 4096/1 | 3.50–99.00 | 8.47 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP2 @1 tok (ms) | 0.0070–0.0140 | 0.0059 | **FAIL** |
| Phase 1 (b) event + backlog | attn_post_proj TP2 @8 tok (ms) | 0.0070–0.0140 | 0.0107 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP2 @64 tok (ms) | 0.0080–0.0160 | 0.0120 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP2 @4096 tok (ms) | 0.0340–0.0400 | 0.0372 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP2 @8192 tok (ms) | 0.0560–0.0620 | 0.0590 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP2 ratio 4096/1 | 2.50–99.00 | 6.28 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP4 @1 tok (ms) | 0.0070–0.0130 | 0.0059 | **FAIL** |
| Phase 1 (b) event + backlog | attn_post_proj TP4 @8 tok (ms) | 0.0070–0.0130 | 0.0082 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP4 @64 tok (ms) | 0.0080–0.0150 | 0.0114 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP4 @4096 tok (ms) | 0.0210–0.0270 | 0.0237 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP4 @8192 tok (ms) | 0.0340–0.0400 | 0.0377 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP4 ratio 4096/1 | 1.70–99.00 | 4.00 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP8 @1 tok (ms) | 0.0070–0.0130 | 0.0059 | **FAIL** |
| Phase 1 (b) event + backlog | attn_post_proj TP8 @8 tok (ms) | 0.0070–0.0130 | 0.0070 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP8 @64 tok (ms) | 0.0070–0.0140 | 0.0106 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP8 @4096 tok (ms) | 0.0160–0.0220 | 0.0191 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP8 @8192 tok (ms) | 0.0230–0.0290 | 0.0265 | PASS |
| Phase 1 (b) event + backlog | attn_post_proj TP8 ratio 4096/1 | 1.30–99.00 | 3.22 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP1 @1 tok (ms) | 0.0500–0.0680 | 0.0314 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP1 @8 tok (ms) | 0.0500–0.0680 | 0.0427 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP1 @64 tok (ms) | 0.0500–0.0680 | 0.0488 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP1 @4096 tok (ms) | 0.0750–0.0850 | 0.0785 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP1 @8192 tok (ms) | 0.0950–0.1050 | 0.0994 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP1 ratio 4096/1 | 0.90–1.35 | 2.50 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP2 @1 tok (ms) | 0.0500–0.0680 | 0.0311 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP2 @8 tok (ms) | 0.0500–0.0680 | 0.0425 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP2 @64 tok (ms) | 0.0500–0.0680 | 0.0503 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP2 @4096 tok (ms) | 0.0600–0.0680 | 0.0639 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP2 @8192 tok (ms) | 0.0780–0.0880 | 0.0823 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP2 ratio 4096/1 | 0.90–1.35 | 2.06 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP4 @1 tok (ms) | 0.0500–0.0680 | 0.0284 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP4 @8 tok (ms) | 0.0500–0.0680 | 0.0422 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP4 @64 tok (ms) | 0.0500–0.0680 | 0.0499 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP4 @4096 tok (ms) | 0.0570–0.0650 | 0.0613 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP4 @8192 tok (ms) | 0.0650–0.0750 | 0.0692 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP4 ratio 4096/1 | 0.90–1.35 | 2.16 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP8 @1 tok (ms) | 0.0500–0.0680 | 0.0311 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP8 @8 tok (ms) | 0.0500–0.0680 | 0.0390 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP8 @64 tok (ms) | 0.0500–0.0680 | 0.0481 | **FAIL** |
| Phase 1 (b) event + backlog | attn_rope TP8 @4096 tok (ms) | 0.0530–0.0610 | 0.0560 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP8 @8192 tok (ms) | 0.0570–0.0670 | 0.0629 | PASS |
| Phase 1 (b) event + backlog | attn_rope TP8 ratio 4096/1 | 0.90–1.35 | 1.80 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP1 @1 tok (ms) | 0.0240–0.0360 | 0.0118 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP1 @8 tok (ms) | 0.0240–0.0360 | 0.0235 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP1 @64 tok (ms) | 0.0260–0.0400 | 0.0249 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP1 @4096 tok (ms) | 0.1760–0.1920 | 0.1839 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP1 @8192 tok (ms) | 0.3280–0.3520 | 0.3354 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP1 ratio 4096/1 | 5.00–99.00 | 15.59 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP2 @1 tok (ms) | 0.0220–0.0340 | 0.0110 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP2 @8 tok (ms) | 0.0220–0.0340 | 0.0227 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP2 @64 tok (ms) | 0.0240–0.0380 | 0.0240 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP2 @4096 tok (ms) | 0.0930–0.1050 | 0.0995 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP2 @8192 tok (ms) | 0.1770–0.1930 | 0.1847 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP2 ratio 4096/1 | 2.80–99.00 | 9.01 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP4 @1 tok (ms) | 0.0220–0.0340 | 0.0110 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP4 @8 tok (ms) | 0.0220–0.0340 | 0.0229 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP4 @64 tok (ms) | 0.0230–0.0360 | 0.0242 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP4 @4096 tok (ms) | 0.0580–0.0680 | 0.0615 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP4 @8192 tok (ms) | 0.0950–0.1070 | 0.1000 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP4 ratio 4096/1 | 1.80–99.00 | 5.61 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP8 @1 tok (ms) | 0.0220–0.0340 | 0.0111 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP8 @8 tok (ms) | 0.0220–0.0340 | 0.0227 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP8 @64 tok (ms) | 0.0230–0.0360 | 0.0228 | **FAIL** |
| Phase 1 (b) event + backlog | attn_pre_proj TP8 @4096 tok (ms) | 0.0430–0.0510 | 0.0459 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP8 @8192 tok (ms) | 0.0610–0.0710 | 0.0666 | PASS |
| Phase 1 (b) event + backlog | attn_pre_proj TP8 ratio 4096/1 | 1.30–99.00 | 4.14 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP1 @1 tok | 3.0–10.0 | 6.2 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP1 @8 tok | 3.0–10.0 | 10.3 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP1 @64 tok | 4.0–12.0 | 12.1 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP1 @4096 tok | 55.0–64.0 | 60.3 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP1 @8192 tok | 88.0–96.0 | 93.5 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP1 ratio 4096/1 | 5.00–99.00 | 9.67 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP2 @1 tok | 3.0–9.0 | 4.0 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP2 @8 tok | 3.0–9.0 | 9.2 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP2 @64 tok | 3.0–10.0 | 10.2 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP2 @4096 tok | 33.0–38.0 | 34.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP2 @8192 tok | 54.0–60.0 | 55.9 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP2 ratio 4096/1 | 3.50–99.00 | 8.68 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP4 @1 tok | 2.0–8.0 | 3.0 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP4 @8 tok | 2.0–8.0 | 6.2 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP4 @64 tok | 3.0–9.0 | 10.1 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP4 @4096 tok | 20.0–26.0 | 21.7 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP4 @8192 tok | 32.0–37.0 | 33.5 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP4 ratio 4096/1 | 2.50–99.00 | 7.23 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP8 @1 tok | 2.0–8.0 | 3.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP8 @8 tok | 2.0–8.0 | 5.5 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP8 @64 tok | 3.0–9.0 | 10.3 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP8 @4096 tok | 11.0–17.0 | 17.8 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP8 @8192 tok | 18.0–24.0 | 23.8 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_post_proj GEMM TP8 ratio 4096/1 | 1.40–99.00 | 5.25 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP1 @1 tok | 25.0–65.0 | 28.2 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP1 @8 tok | 25.0–65.0 | 40.8 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP1 @64 tok | 25.0–65.0 | 49.3 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP1 @4096 tok | 70.0–90.0 | 82.8 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP1 @8192 tok | 90.0–110.0 | 103.5 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP1 ratio 4096/1 | 0.90–2.50 | 2.93 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP2 @1 tok | 25.0–65.0 | 24.4 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP2 @8 tok | 25.0–65.0 | 35.0 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP2 @64 tok | 25.0–65.0 | 46.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP2 @4096 tok | 55.0–70.0 | 64.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP2 @8192 tok | 80.0–95.0 | 82.7 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP2 ratio 4096/1 | 0.90–2.50 | 2.63 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP4 @1 tok | 25.0–65.0 | 22.0 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP4 @8 tok | 25.0–65.0 | 37.0 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP4 @64 tok | 25.0–65.0 | 44.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP4 @4096 tok | 50.0–65.0 | 62.0 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP4 @8192 tok | 60.0–75.0 | 66.7 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP4 ratio 4096/1 | 0.90–2.50 | 2.82 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP8 @1 tok | 25.0–65.0 | 26.2 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP8 @8 tok | 25.0–65.0 | 34.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP8 @64 tok | 25.0–65.0 | 46.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP8 @4096 tok | 45.0–62.0 | 61.9 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP8 @8192 tok | 55.0–68.0 | 68.6 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_rope 13-kernel sum TP8 ratio 4096/1 | 0.90–2.50 | 2.37 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP1 @1 tok | 15.0–35.0 | 12.0 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP1 @8 tok | 15.0–35.0 | 23.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP1 @64 tok | 16.0–38.0 | 26.1 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP1 @4096 tok | 170.0–190.0 | 181.2 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP1 @8192 tok | 320.0–350.0 | 332.7 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP1 ratio 4096/1 | 4.50–99.00 | 15.10 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP2 @1 tok | 14.0–32.0 | 9.4 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP2 @8 tok | 14.0–32.0 | 20.8 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP2 @64 tok | 15.0–35.0 | 22.5 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP2 @4096 tok | 88.0–100.0 | 96.9 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP2 @8192 tok | 175.0–190.0 | 182.3 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP2 ratio 4096/1 | 2.70–99.00 | 10.31 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP4 @1 tok | 14.0–32.0 | 8.6 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP4 @8 tok | 14.0–32.0 | 21.0 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP4 @64 tok | 15.0–35.0 | 23.2 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP4 @4096 tok | 55.0–65.0 | 61.2 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP4 @8192 tok | 92.0–104.0 | 98.7 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP4 ratio 4096/1 | 1.70–99.00 | 7.08 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP8 @1 tok | 14.0–32.0 | 9.2 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP8 @8 tok | 14.0–32.0 | 22.8 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP8 @64 tok | 15.0–35.0 | 23.9 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP8 @4096 tok | 40.0–48.0 | 49.6 | **FAIL** |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP8 @8192 tok | 58.0–66.0 | 65.4 | PASS |
| Phase 1 (c) kernel trace (µs) | attn_pre_proj 5-kernel sum TP8 ratio 4096/1 | 1.25–99.00 | 5.40 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP1 @8 tok | 3.00–12.00 | 3.46 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP1 @64 tok | 3.00–12.00 | 3.15 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP2 @8 tok | 3.00–12.00 | 3.75 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP2 @64 tok | 3.00–12.00 | 3.59 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP4 @8 tok | 3.00–12.00 | 5.92 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP4 @64 tok | 3.00–12.00 | 3.75 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP8 @8 tok | 3.00–12.00 | 6.89 | PASS |
| Phase 1 (a)/(c) split | attn_post_proj event/kernel TP8 @64 tok | 3.00–12.00 | 3.75 | PASS |
| Phase 2 closure | TP2 @3072 W_a (ms) | 0.440–0.600 | 0.624 | **FAIL** |
| Phase 2 closure | TP2 @3072 W_c traced period (ms) | 0.650–0.850 | 0.737 | PASS |
| Phase 2 closure | TP2 @3072 G_sum (ms) | 0.300–0.360 | 0.312 | PASS |
| Phase 2 closure | TP2 @3072 G₃ (ms) | 0.150–0.180 | 0.164 | PASS |
| Phase 2 closure | TP2 @3072 E₃ (ms) | 0.220–0.270 | 0.251 | PASS |
| Phase 2 closure | TP2 @3072 B₃ (ms) | 0.160–0.190 | 0.171 | PASS |
| Phase 2 closure | TP2 @3072 idle = 1−G_sum/W_a (%) | 25–45 | 50 | **FAIL** |
| Phase 2 closure | TP2 @3072 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.171 ≤ 0.312 ≤ 0.624 | PASS |
| Phase 2 closure | TP2 @3072 check B: E/G attn_pre_proj | 1.30–1.70 | 1.45 | PASS |
| Phase 2 closure | TP2 @3072 check B: E/G attn_rope | 1.20–1.50 | 1.42 | PASS |
| Phase 2 closure | TP2 @3072 check B: E/G attn_post_proj | 1.70–2.30 | 1.98 | PASS |
| Phase 2 closure | TP2 @3072 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +3.4 | **FAIL** |
| Phase 2 closure | TP2 @3072 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -6.2 | **FAIL** |
| Phase 2 closure | TP2 @3072 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +3.0 | **FAIL** |
| Phase 2 closure | TP2 @3072 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 1.004 | PASS |
| Phase 2 closure | TP2 @3072 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.001 | PASS |
| Phase 2 closure | TP2 @3072 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.15 | **FAIL** |
| Phase 2 closure | TP2 @3072 traced sleep (ms) vs requested 100 | 100.0–100.0 | 77.4 | **FAIL** |
| Phase 2 closure | TP2 @4192 W_a (ms) | 0.440–0.600 | 0.626 | **FAIL** |
| Phase 2 closure | TP2 @4192 W_c traced period (ms) | 0.650–0.850 | 0.742 | PASS |
| Phase 2 closure | TP2 @4192 G_sum (ms) | 0.350–0.410 | 0.378 | PASS |
| Phase 2 closure | TP2 @4192 G₃ (ms) | 0.170–0.200 | 0.205 | **FAIL** |
| Phase 2 closure | TP2 @4192 E₃ (ms) | 0.210–0.270 | 0.262 | PASS |
| Phase 2 closure | TP2 @4192 B₃ (ms) | 0.180–0.210 | 0.212 | **FAIL** |
| Phase 2 closure | TP2 @4192 idle = 1−G_sum/W_a (%) | 15–40 | 40 | PASS |
| Phase 2 closure | TP2 @4192 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.212 ≤ 0.378 ≤ 0.626 | PASS |
| Phase 2 closure | TP2 @4192 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +3.3 | **FAIL** |
| Phase 2 closure | TP2 @4192 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -4.8 | PASS |
| Phase 2 closure | TP2 @4192 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +4.3 | PASS |
| Phase 2 closure | TP2 @4192 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 1.003 | PASS |
| Phase 2 closure | TP2 @4192 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.003 | PASS |
| Phase 2 closure | TP2 @4192 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.14 | **FAIL** |
| Phase 2 closure | TP2 @4192 traced sleep (ms) vs requested 100 | 100.0–100.0 | 77.3 | **FAIL** |
| Phase 2 closure | TP2 @6144 W_a (ms) | 0.440–0.600 | 0.628 | **FAIL** |
| Phase 2 closure | TP2 @6144 W_c traced period (ms) | 0.650–0.850 | 0.744 | PASS |
| Phase 2 closure | TP2 @6144 G_sum (ms) | 0.470–0.530 | 0.465 | **FAIL** |
| Phase 2 closure | TP2 @6144 G₃ (ms) | 0.240–0.280 | 0.247 | PASS |
| Phase 2 closure | TP2 @6144 E₃ (ms) | 0.240–0.300 | 0.272 | PASS |
| Phase 2 closure | TP2 @6144 B₃ (ms) | 0.250–0.290 | 0.250 | PASS |
| Phase 2 closure | TP2 @6144 idle = 1−G_sum/W_a (%) | 0–20 | 26 | **FAIL** |
| Phase 2 closure | TP2 @6144 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.250 ≤ 0.465 ≤ 0.628 | PASS |
| Phase 2 closure | TP2 @6144 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +2.8 | **FAIL** |
| Phase 2 closure | TP2 @6144 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -2.8 | PASS |
| Phase 2 closure | TP2 @6144 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +4.9 | PASS |
| Phase 2 closure | TP2 @6144 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 0.999 | PASS |
| Phase 2 closure | TP2 @6144 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.023 | PASS |
| Phase 2 closure | TP2 @6144 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.13 | **FAIL** |
| Phase 2 closure | TP2 @6144 traced sleep (ms) vs requested 100 | 100.0–100.0 | 77.4 | **FAIL** |
| Phase 2 closure | TP2 @8192 W_a = max(0.44–0.60, G_sum) ±10 % | 0.554–0.677 | 0.677 | **FAIL** |
| Phase 2 closure | TP2 @8192 W_c traced period (ms) | 0.700–0.900 | 0.754 | PASS |
| Phase 2 closure | TP2 @8192 G_sum (ms) | 0.600–0.680 | 0.615 | PASS |
| Phase 2 closure | TP2 @8192 G₃ (ms) | 0.310–0.350 | 0.321 | PASS |
| Phase 2 closure | TP2 @8192 E₃ (ms) | 0.310–0.360 | 0.327 | PASS |
| Phase 2 closure | TP2 @8192 B₃ (ms) | 0.320–0.360 | 0.326 | PASS |
| Phase 2 closure | TP2 @8192 idle = 1−G_sum/W_a (%) | 0–8 | 9 | **FAIL** |
| Phase 2 closure | TP2 @8192 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.326 ≤ 0.615 ≤ 0.677 | PASS |
| Phase 2 closure | TP2 @8192 check B: E/G attn_pre_proj | 0.95–1.08 | 1.01 | PASS |
| Phase 2 closure | TP2 @8192 check B: E/G attn_rope | 0.95–1.08 | 1.00 | PASS |
| Phase 2 closure | TP2 @8192 check B: E/G attn_post_proj | 0.95–1.08 | 1.06 | PASS |
| Phase 2 closure | TP2 @8192 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +2.7 | **FAIL** |
| Phase 2 closure | TP2 @8192 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -2.1 | PASS |
| Phase 2 closure | TP2 @8192 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +2.3 | **FAIL** |
| Phase 2 closure | TP2 @8192 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 0.993 | PASS |
| Phase 2 closure | TP2 @8192 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.000 | PASS |
| Phase 2 closure | TP2 @8192 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.08 | PASS |
| Phase 2 closure | TP2 @8192 traced sleep (ms) vs requested 100 | 100.0–100.0 | 77.5 | **FAIL** |
| Phase 2 closure | TP4 @3072 W_a (ms) | 0.440–0.600 | 0.610 | **FAIL** |
| Phase 2 closure | TP4 @3072 W_c traced period (ms) | 0.650–0.850 | 0.757 | PASS |
| Phase 2 closure | TP4 @3072 G_sum (ms) | 0.260–0.310 | 0.267 | PASS |
| Phase 2 closure | TP4 @3072 G₃ (ms) | 0.110–0.130 | 0.127 | PASS |
| Phase 2 closure | TP4 @3072 E₃ (ms) | 0.200–0.260 | 0.236 | PASS |
| Phase 2 closure | TP4 @3072 B₃ (ms) | 0.120–0.140 | 0.129 | PASS |
| Phase 2 closure | TP4 @3072 idle = 1−G_sum/W_a (%) | 35–55 | 56 | **FAIL** |
| Phase 2 closure | TP4 @3072 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.129 ≤ 0.267 ≤ 0.610 | PASS |
| Phase 2 closure | TP4 @3072 check B: E/G attn_pre_proj | 1.30–1.70 | 1.70 | PASS |
| Phase 2 closure | TP4 @3072 check B: E/G attn_rope | 1.20–1.50 | 1.77 | **FAIL** |
| Phase 2 closure | TP4 @3072 check B: E/G attn_post_proj | 1.70–2.30 | 2.56 | **FAIL** |
| Phase 2 closure | TP4 @3072 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +2.9 | **FAIL** |
| Phase 2 closure | TP4 @3072 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -6.2 | **FAIL** |
| Phase 2 closure | TP4 @3072 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +0.4 | **FAIL** |
| Phase 2 closure | TP4 @3072 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 1.052 | **FAIL** |
| Phase 2 closure | TP4 @3072 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.031 | PASS |
| Phase 2 closure | TP4 @3072 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.24 | **FAIL** |
| Phase 2 closure | TP4 @3072 traced sleep (ms) vs requested 100 | 100.0–100.0 | 78.0 | **FAIL** |
| Phase 2 closure | TP4 @4192 W_a (ms) | 0.440–0.600 | 0.614 | **FAIL** |
| Phase 2 closure | TP4 @4192 W_c traced period (ms) | 0.650–0.850 | 0.756 | PASS |
| Phase 2 closure | TP4 @4192 G_sum (ms) | 0.290–0.340 | 0.308 | PASS |
| Phase 2 closure | TP4 @4192 G₃ (ms) | 0.120–0.140 | 0.147 | **FAIL** |
| Phase 2 closure | TP4 @4192 E₃ (ms) | 0.200–0.260 | 0.237 | PASS |
| Phase 2 closure | TP4 @4192 B₃ (ms) | 0.130–0.150 | 0.150 | **FAIL** |
| Phase 2 closure | TP4 @4192 idle = 1−G_sum/W_a (%) | 30–50 | 50 | PASS |
| Phase 2 closure | TP4 @4192 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.150 ≤ 0.308 ≤ 0.614 | PASS |
| Phase 2 closure | TP4 @4192 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +2.4 | **FAIL** |
| Phase 2 closure | TP4 @4192 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -5.4 | **FAIL** |
| Phase 2 closure | TP4 @4192 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +0.2 | **FAIL** |
| Phase 2 closure | TP4 @4192 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 1.021 | PASS |
| Phase 2 closure | TP4 @4192 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.002 | PASS |
| Phase 2 closure | TP4 @4192 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.20 | **FAIL** |
| Phase 2 closure | TP4 @4192 traced sleep (ms) vs requested 100 | 100.0–100.0 | 77.9 | **FAIL** |
| Phase 2 closure | TP4 @6144 W_a (ms) | 0.440–0.600 | 0.616 | **FAIL** |
| Phase 2 closure | TP4 @6144 W_c traced period (ms) | 0.650–0.850 | 0.751 | PASS |
| Phase 2 closure | TP4 @6144 G_sum (ms) | 0.360–0.410 | 0.359 | **FAIL** |
| Phase 2 closure | TP4 @6144 G₃ (ms) | 0.140–0.170 | 0.166 | PASS |
| Phase 2 closure | TP4 @6144 E₃ (ms) | 0.200–0.260 | 0.242 | PASS |
| Phase 2 closure | TP4 @6144 B₃ (ms) | 0.150–0.180 | 0.171 | PASS |
| Phase 2 closure | TP4 @6144 idle = 1−G_sum/W_a (%) | 20–40 | 42 | **FAIL** |
| Phase 2 closure | TP4 @6144 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.171 ≤ 0.359 ≤ 0.616 | PASS |
| Phase 2 closure | TP4 @6144 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +2.9 | **FAIL** |
| Phase 2 closure | TP4 @6144 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -4.5 | PASS |
| Phase 2 closure | TP4 @6144 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +2.8 | **FAIL** |
| Phase 2 closure | TP4 @6144 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 1.016 | PASS |
| Phase 2 closure | TP4 @6144 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.003 | PASS |
| Phase 2 closure | TP4 @6144 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.19 | **FAIL** |
| Phase 2 closure | TP4 @6144 traced sleep (ms) vs requested 100 | 100.0–100.0 | 77.9 | **FAIL** |
| Phase 2 closure | TP4 @8192 W_a (ms) | 0.440–0.600 | 0.629 | **FAIL** |
| Phase 2 closure | TP4 @8192 W_c traced period (ms) | 0.650–0.850 | 0.769 | PASS |
| Phase 2 closure | TP4 @8192 G_sum (ms) | 0.430–0.480 | 0.439 | PASS |
| Phase 2 closure | TP4 @8192 G₃ (ms) | 0.170–0.200 | 0.199 | PASS |
| Phase 2 closure | TP4 @8192 E₃ (ms) | 0.210–0.270 | 0.244 | PASS |
| Phase 2 closure | TP4 @8192 B₃ (ms) | 0.180–0.210 | 0.207 | PASS |
| Phase 2 closure | TP4 @8192 idle = 1−G_sum/W_a (%) | 10–30 | 30 | **FAIL** |
| Phase 2 closure | TP4 @8192 check A: B₃ ≤ G_sum ≤ W_a | identity | 0.207 ≤ 0.439 ≤ 0.629 | PASS |
| Phase 2 closure | TP4 @8192 check B: E/G attn_pre_proj | 1.00–1.15 | 1.14 | PASS |
| Phase 2 closure | TP4 @8192 check B: E/G attn_rope | 0.95–1.10 | 1.00 | PASS |
| Phase 2 closure | TP4 @8192 check B: E/G attn_post_proj | 1.50–1.90 | 1.94 | **FAIL** |
| Phase 2 closure | TP4 @8192 check A′: B_op − G_op(cb) attn_post_proj (µs) | +4.0–+12.0 | +6.0 | PASS |
| Phase 2 closure | TP4 @8192 check A′: B_op − G_op(cb) attn_rope (µs) | -5.0–+10.0 | -1.9 | PASS |
| Phase 2 closure | TP4 @8192 check A′: B_op − G_op(cb) attn_pre_proj (µs) | +4.0–+12.0 | +2.6 | **FAIL** |
| Phase 2 closure | TP4 @8192 check D: no-knob/knob kernel o_proj GEMM | 0.950–1.050 | 1.057 | **FAIL** |
| Phase 2 closure | TP4 @8192 check D: no-knob/knob kernel QKV scope | 0.950–1.050 | 1.013 | PASS |
| Phase 2 closure | TP4 @8192 check E: W′/G_sum(cb) (sleep measured, not 100 ms) | 0.90–1.10 | 1.16 | **FAIL** |
| Phase 2 closure | TP4 @8192 traced sleep (ms) vs requested 100 | 100.0–100.0 | 78.0 | **FAIL** |