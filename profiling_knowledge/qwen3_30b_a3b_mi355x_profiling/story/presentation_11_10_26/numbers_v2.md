# numbers_v2.md - recomputed 2026-10-11 after independent verification

Supersedes `numbers.md`. See `../review/FINDINGS_LOG.md`. Originals kept for comparison.

## Step naming (corrected)

Job 21356 is the X1 backlog-knob DIAGNOSTIC, not an iteration. The dataset iterations are 21313 -> 21334 -> 21361 -> 21483 -> 21500 -> 21519 -> 21539. v1's 'there is no it 4' was wrong (it 4 = job 21519).

| job | role | in git? | md5 |
|---|---|---|---|
| 21313 | original dense grid | yes | `97d6255414` |
| 21334 | rerun, different node | yes | `3986255253` |
| 21361 | GC fix | yes | `9381d3edaa` |
| 21483 | two-column + fused RoPE | **no - temp dir** | `1d84b3d195` |
| 21500 | batch-flush fix | **no - temp dir** | `cd405d6428` |
| 21519 | gemm_alloc backlog, settle 3 | **no - temp dir** | `7016bd6260` |
| 21539 | FINAL - settle 8 | yes | `3635e4d305` |
| 21356 | X1 backlog-knob diagnostic *(diagnostic, not an iteration)* | **no - temp dir** | `93704fef54` |

## Definition

Per shared num_tokens, ratio = job_A median / job_B median; then the median over tokens. min/max/percentiles are over the same per-token ratios. Labels are time_stats.<op>.median in ms.

## Original (21313) / final (21539)

### attn_pre_proj

| TP | range | n | median | min | p05 | p25 | p75 | p95 | max | ratio<1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | <=2048 | 2048 | **1.53** | 1.27 | 1.33 | 1.41 | 1.99 | 3.12 | 4.67 | 0 |
| 1 | <=5000 | 2416 | **1.48** | 0.98 | 1.02 | 1.35 | 1.84 | 3.05 | 4.67 | 38 |
| 1 | full | 3327 | **1.38** | 0.95 | 0.99 | 1.02 | 1.62 | 2.89 | 4.67 | 352 |
| 2 | <=2048 | 2048 | **2.12** | 1.56 | 1.63 | 1.80 | 2.87 | 3.78 | 5.39 | 0 |
| 2 | <=5000 | 2416 | **2.01** | 1.20 | 1.31 | 1.68 | 2.71 | 3.73 | 5.39 | 0 |
| 2 | full | 3327 | **1.73** | 0.96 | 1.00 | 1.19 | 2.32 | 3.55 | 5.39 | 230 |
| 4 | <=2048 | 2048 | **2.86** | 1.99 | 2.14 | 2.47 | 3.35 | 4.10 | 5.50 | 0 |
| 4 | <=5000 | 2416 | **2.73** | 1.32 | 1.49 | 2.23 | 3.25 | 4.07 | 5.50 | 0 |
| 4 | full | 3327 | **2.36** | 0.97 | 1.00 | 1.33 | 3.05 | 3.97 | 5.50 | 170 |
| 8 | <=2048 | 2048 | **3.25** | 2.56 | 2.72 | 2.96 | 3.61 | 4.07 | 5.66 | 0 |
| 8 | <=5000 | 2416 | **3.14** | 1.49 | 1.81 | 2.81 | 3.55 | 4.04 | 5.66 | 0 |
| 8 | full | 3327 | **2.87** | 0.95 | 1.00 | 1.46 | 3.39 | 3.97 | 5.66 | 187 |

### attn_post_proj

| TP | range | n | median | min | p05 | p25 | p75 | p95 | max | ratio<1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | <=2048 | 2048 | **1.81** | 0.97 | 1.24 | 1.67 | 2.07 | 2.70 | 3.21 | 5 |
| 1 | <=5000 | 2416 | **1.76** | 0.97 | 1.01 | 1.49 | 2.01 | 2.61 | 3.21 | 69 |
| 1 | full | 3327 | **1.61** | 0.96 | 1.00 | 1.03 | 1.89 | 2.52 | 3.21 | 222 |
| 2 | <=2048 | 2048 | **2.30** | 1.90 | 2.03 | 2.12 | 2.71 | 3.21 | 3.56 | 0 |
| 2 | <=5000 | 2416 | **2.18** | 0.99 | 1.54 | 2.07 | 2.64 | 3.17 | 3.56 | 4 |
| 2 | full | 3327 | **2.10** | 0.93 | 1.00 | 1.07 | 2.48 | 3.09 | 3.56 | 122 |
| 4 | <=2048 | 2048 | **2.85** | 2.45 | 2.55 | 2.69 | 3.12 | 3.42 | 4.74 | 0 |
| 4 | <=5000 | 2416 | **2.79** | 1.69 | 2.06 | 2.61 | 3.08 | 3.39 | 4.74 | 0 |
| 4 | full | 3327 | **2.65** | 0.97 | 1.01 | 1.89 | 2.96 | 3.35 | 4.74 | 69 |
| 8 | <=2048 | 2048 | **3.08** | 2.57 | 2.75 | 2.93 | 3.31 | 3.75 | 5.50 | 0 |
| 8 | <=5000 | 2416 | **3.03** | 1.86 | 2.35 | 2.80 | 3.26 | 3.73 | 5.50 | 0 |
| 8 | full | 3327 | **2.88** | 0.96 | 1.03 | 2.17 | 3.15 | 3.68 | 5.50 | 41 |

### attn_rope

| TP | range | n | median | min | p05 | p25 | p75 | p95 | max | ratio<1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | <=2048 | 2048 | **7.35** | 3.81 | 4.04 | 4.86 | 10.67 | 12.63 | 13.80 | 0 |
| 1 | <=5000 | 2416 | **6.41** | 2.65 | 2.96 | 4.28 | 10.17 | 12.58 | 13.80 | 0 |
| 1 | full | 3327 | **4.50** | 1.82 | 1.98 | 2.63 | 8.83 | 12.36 | 13.80 | 0 |
| 2 | <=2048 | 2048 | **11.55** | 7.58 | 8.67 | 9.95 | 13.31 | 14.89 | 15.78 | 0 |
| 2 | <=5000 | 2416 | **10.89** | 3.50 | 4.01 | 9.11 | 12.96 | 14.83 | 15.78 | 0 |
| 2 | full | 3327 | **9.67** | 2.09 | 2.29 | 3.35 | 12.52 | 14.63 | 15.78 | 0 |
| 4 | <=2048 | 2048 | **13.46** | 10.36 | 11.34 | 12.62 | 14.09 | 15.31 | 15.89 | 0 |
| 4 | <=5000 | 2416 | **13.21** | 5.67 | 7.67 | 11.87 | 14.01 | 15.21 | 15.89 | 0 |
| 4 | full | 3327 | **12.37** | 2.34 | 2.67 | 5.38 | 13.75 | 14.99 | 15.89 | 0 |
| 8 | <=2048 | 2048 | **13.54** | 12.31 | 12.80 | 13.12 | 14.28 | 15.09 | 15.74 | 0 |
| 8 | <=5000 | 2416 | **13.44** | 9.37 | 10.88 | 12.92 | 14.18 | 15.07 | 15.74 | 0 |
| 8 | full | 3327 | **13.02** | 3.54 | 3.97 | 9.44 | 13.77 | 15.01 | 15.74 | 0 |

> Jobs 21313/21334/21361/21356 timed a torch fallback that is not RoPE; 21483 onward use the fused kernel. The original/final rope ratio mixes a kernel change with the timing change.

## Which later step actually moved the medians?

later / earlier, pooled over TP, full grid:

| transition | attn_pre_proj | attn_post_proj | attn_rope |
|---|---|---|---|
| 21483->21500 (batch-flush fix) | 0.9990 | 0.9992 | 1.0000 |
| 21500->21519 (gemm_alloc backlog + settle 3) | 0.9769 | 0.9755 | 0.9845 |
| 21519->21539 (settle 8) | 1.0005 | 1.0017 | 1.0000 |

Only the `gemm_alloc` backlog change moved the medians (1.5-2.4 % down, all three ops). The batch-flush fix and settle-8 left them within 0.2 %.

## Controls across the RoPE change (job 21356 / job 21483, median per TP)

| op | TP1 | TP2 | TP4 | TP8 |
|---|---|---|---|---|
| attn_pre_proj | 0.988 | 0.989 | 0.989 | 0.994 |
| attn_post_proj | 1.003 | 1.002 | 0.999 | 0.997 |
| attn_rope | 2.585 | 3.440 | 4.447 | 6.063 |

The GEMM scopes agree within 1.2 % across that protocol change while attn_rope moves 2.6-6.1x. That is what makes them controls - a measured agreement, not an assumption from code identity.

## Deck claims vs data

| claim | op | range | computed median ratio across TPs | consistent (+/-5 %) |
|---|---|---|---|---|
| GEMM 1.3-2.9x | attn_pre_proj | <=2048 | 1.53-3.25 | **NO** |
| GEMM 1.3-2.9x | attn_pre_proj | <=5000 | 1.48-3.14 | **NO** |
| GEMM 1.3-2.9x | attn_pre_proj | full | 1.38-2.87 | yes |
| GEMM 1.3-2.9x | attn_post_proj | <=2048 | 1.81-3.08 | **NO** |
| GEMM 1.3-2.9x | attn_post_proj | <=5000 | 1.76-3.03 | yes |
| GEMM 1.3-2.9x | attn_post_proj | full | 1.61-2.88 | yes |
| RoPE 4-12x | attn_rope | <=2048 | 7.35-13.54 | **NO** |
| RoPE 4-12x | attn_rope | <=5000 | 6.41-13.44 | **NO** |
| RoPE 4-12x | attn_rope | full | 4.5-13.02 | **NO** |

## GC spike (corrected)

| job | TP1 | TP2 | TP4 | TP8 | overall |
|---|---|---|---|---|---|
| 21313 | 150.4-242.8 | 151.0-255.1 | 156.4-268.6 | 152.9-263.9 | **150.4-268.6 ms** |
| 21334 | 149.7-262.9 | 151.6-266.0 | 146.4-287.7 | 148.5-244.5 | **146.4-287.7 ms** |

v1's `numbers.md` reported 150-243 ms, which is TP1 of job 21313 only. The page's Chapter 2 '150-263 ms' for job 21334 is wrong.

- s05 shaded region: 2-6,704 tokens (501-point rolling median, 0.0941 -> 0.0737 ms).
