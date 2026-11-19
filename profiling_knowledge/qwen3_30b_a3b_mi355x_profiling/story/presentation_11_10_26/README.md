# presentation_11_10_26 — figures for the Qwen3-30B-A3B / MI355X profiling story

Everything here is produced by one script: `python make_presentation_figures.py` (run from anywhere). It reads the same data as
`../evolution/evolution.ipynb` (config cell exec'd from the notebook; source CSV if present, else `../evolution/cache/`), is read-only on
all data, and writes only into this folder. Files: 8 PNGs, `numbers.md` / `numbers.json` (ratio verification), `manifest.json` (per-figure
steps, source paths, md5, y-limits, caveats). Every curve is the per-shape **median** (`time_stats.<op>.median`), unsmoothed, except s04.

Common style: linear y from 0, white background, no figure title, x = num_tokens, y = median time (ms), TP colours from the notebook
(TP 1 blue, TP 2 orange, TP 4 green, TP 8 amber), older step light / final dark and thicker, fonts 12 pt, dpi 200.
Single-panel 6.5x3.8 in, two-panel 10x3.8 in (the appendix is 10x7.6 in, 2x2 panels).

## Step mapping (please read)
The brief names "original, it 1 ... it 4, final". The notebook has these full datasets, and this is how they are named here:

| name in deck | notebook id | job | what it is |
|---|---|---|---|
| original | D1 | 21313 | first re-collection; host-bound time; RoPE = torch fallback |
| it 1 (GC removed) | D2 | 21361 | host-bound; RoPE = torch fallback |
| it 2 (GPU-bound) | D3 | 21356 | GPU-bound; RoPE = torch fallback; **tokens >= 1,024 only** |
| it 3 (RoPE fix) | D4 | 21483 | GPU-bound; fused RoPE |
| final | D5 | 21539 | GPU-bound; fused RoPE; run-position fixes |

There is no "it 4": after the original there are only four steps (it 1-3 + final). The brief also calls the torch-fallback step "it 1" (slide 6)
and the GPU-bound step "it 2" (slide 5); in the notebook the first GPU-bound step *is* the torch-fallback one (D3), so s06 uses D3 vs D4.
D0 (inherited, <= 4,096 tokens, 20 reps) is not used.

## Figures
- **s02_pre_proj_original.png** (slide 2) — `attn_pre_proj` in the original data at TP 1/2/4/8, full range and <= 2,048 tokens. Shows that the
  curves do not behave like "more tokens, more time": at <= 2,048 tokens TP 2/4/8 are flat/noisy at ~0.09 ms and TP 8 even falls in the full range.
  Does not show the cause. Each panel has its own y-range (the zoom would be unreadable on the full-range scale), matched to s03 panel by panel.
- **s03_pre_proj_final_all_tp.png** (slide 3) — the same op and panels from the final dataset: smooth rising curves ordered by TP, with the 15-30 %
  tile-selection steps that the notebook says are real hardware behaviour. Same y-limits as s02, panel by panel, so the two can be compared by eye.
- **s04_gc_spike_per_repetition.png** (slide 4) — TP 1, original data: min-max band over the 50 timed runs, median, and the 8 token counts
  (1,968-1,975) whose max is 150-243 ms (~1,000x the median). **Exception to the global style:** log y. It is built from the stored per-shape
  min/max/median (per-repetition samples are not kept for this dataset), not from individual runs. It shows that the median hides the spike; it does not show the GC cause.
- **s05_pre_proj_tp8_hostbound_vs_gpubound.png** (slide 5) — TP 8: "it 1" (host-bound, light) falls from ~0.094 to ~0.074 ms while "it 2" (GPU-bound, dark)
  rises; they meet near 8,000 tokens and agree above that. The shaded region (2-6,704 tokens) is computed, not hand-drawn: argmax -> argmin of a 501-point rolling
  median of the host-bound curve (detection only; the curves are raw). The old slide shaded 4,000-6,000; the decline in the data is gradual and starts at the lowest token counts.
  "it 2" starts at 1,024 tokens (no data below).
- **s06_rope_fallback_vs_fused.png** (slide 6) — `attn_rope`, TP 1, <= 2,048 tokens: the torch fallback (D3, GPU-bound, 1,024-2,048 only) is ~4-5x slower than the fused kernel (D4).
  The note on the figure says it is a different computation. It does not show that the fused kernel is *correct*, and it does not use the host-bound fallback data (full 1-2,048 range but a different quantity).
- **s07_pre_proj_before_after.png** (slide 7) — `attn_pre_proj`, TP 1 and TP 8, <= 2,048 tokens, original vs final, same y-limits in both panels.
  TP 8: ~0.09 -> ~0.023-0.033 ms. **s07_pre_proj_before_after_full.png** is the same at full range (same y in both panels, so TP 8 looks compressed; the curves agree above ~8,000 tokens).
- **appendix_iteration_ratio_to_final.png** (diagnostic) — `attn_pre_proj`, per TP: median time of each earlier step divided by the final step, per token, log y.
  Job numbers are in the legend. Original and it 1 overlap (both host-bound); it 2 and it 3 sit within a few percent of 1 (see numbers.md: median ratio 1.02-1.03, max |ratio-1| 0.05-0.09), i.e. iterations 3-5 changed `attn_pre_proj` only slightly. It does not say anything about the other ops.

## numbers.md — main findings (details and exact definitions inside)
- Definition: per-token ratio (step median / final median on shared token counts), then the median over tokens; the ratio of medians is reported alongside.
- GEMM "1.3-2.9x": matches `attn_pre_proj` over the **full range** (1.38-2.87 across TPs) and `attn_post_proj` (1.61-2.88). Over <= 2,048 tokens it is larger (pre_proj 1.53-3.25, post_proj 1.81-3.08; TP 8 pre_proj 3.25x, consistent with the ~3-4x visible in s07).
- RoPE "4-12x": original/final is 4.5-13.0x (full range), 7.4-13.5x (<= 2,048); it also mixes a kernel change with timing changes.
- Iterations 3+ for `attn_pre_proj`: it 3 / final median ratio 1.02-1.03, max |ratio-1| 0.06-0.09.
