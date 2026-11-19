# CORRECTIONS_TO_EXISTING — errors found in the published page, STORY.md, the notebook and the presentation set

Prepared 2026-10-11, Phase 1. **No existing file was edited and the Confluence page was not touched.** Line numbers for
the live page refer to the markdown rendering retrieved on 2026-10-11 (page id 7166984214, lastModified Sept 24 2026).
`STORY.md` line numbers are as of HEAD `25e163c`.

Severity: **A** = factually wrong, must change · **B** = unsupported or overstated, must be qualified · **C** = unclear
or inconsistent, should be clarified.

## 1. Live Confluence page (id 7166984214) and STORY.md

These two carry the same text; the page was generated from `STORY.md` at commit `4d18e8d`.

| # | Sev | Location | Wrong text | Corrected text |
|---|---|---|---|---|
| 1 | **A** | page §0 line 11; `STORY.md:29` | "the new collections are **monotonic**, instrument-accounted…" | "the new collections rise with token count apart from step structure and measurement scatter: 43–48 % of consecutive-token steps still decrease, with a median down-step of ≈1.1 % and isolated steps as large as −42 % (`attn_post_proj` TP8, 12288→12304). Monotonicity was never an acceptance criterion; the dip diagnosis rests on the X1 intervention (job 21356) and the rocprofv3 traces." |
| 2 | **A** | page line 93; `STORY.md:117` | job 21334: "timed run 11, **150–263 ms**" | "timed run 11, **146.4–287.7 ms** across TP 1/2/4/8 (TP8 alone is 148.5–244.5 ms)" |
| 3 | **B** | page §0 line 11; `STORY.md:29-30` | "agree with kernel traces to a **constant ≈3 µs** per scope boundary" | "the GPU-bound event value exceeds the traced kernel duration by **+1.4 to +5.4 µs** (median +2.35 µs) over the eight TP2 `attn_post_proj` shapes that were compared. The comparison uses eight hard-coded, 0.1-µs-rounded trace values from job 21376 against event points from job 21400; it is not a constant and has not been established beyond that one op/TP." |
| 4 | **B** | page §0 line 11; `STORY.md:29` | "overstated GEMM costs by **1.3–2.9×** and RoPE by **4–12×**" | "overstated GEMM costs by 1.38–2.87× (`attn_pre_proj`) and 1.61–2.88× (`attn_post_proj`) as a full-grid per-TP median, rising to 3.25× and 3.08× for tokens ≤2,048; and `attn_rope` by **4.50–13.02×** full-grid. 1–11 % of shapes run the other way (352 `attn_pre_proj` TP1 shapes have ratio <1). The RoPE figure mixes a kernel change with the timing change and is not a pure instrument effect." |
| 5 | **B** | `14_regressor_dataset_handoff.md:110-111` (not on the page, but it is the handoff the page points to) | "`T2 TP4 … = 1.972` … **Known flake item**, reported not re-thresholded" | "T2 TP4 = **1.9724 in both job 21519 and job 21539** — two collections, different days, different settle protocols, agreeing to four decimals. It is a reproducible systematic result, not a flake. It appeared when the `gemm_alloc` backlog lowered medians ≈2.4 %. Separately, T2 at **TP8 is below 2.0 in every post-fix dataset** (1.76–1.82) but the validator only tests TP 1, 2, 4." |
| 6 | **C** | page line 65; `STORY.md:93` | "It was always **timed run 11** (absolute sample 14, i.e. the 15th forward including 3 warm-ups)" | Keep the facts but state the base: "…0-based timed run 11 (= the 12th timed run counting from 1), absolute sample index 14 (0-based), i.e. the 15th forward including 3 warm-ups." The page discloses the 0-based convention only in Fig. 4's caption; Chapter 1 does not, and `evolution.ipynb` plots a 1-based axis. |
| 7 | **B** | `14_regressor_dataset_handoff.md:100` | "Per-row relative MAD of the 25 samples is **~0.55 %**" | "0.54–0.56 % for `attn_pre_proj`; 0.68–0.92 % for `attn_post_proj` (rising with TP); 0.61–0.77 % for `attn_rope`; 0.76–0.78 % for the norms; **2.47 % for `emb`**." |
| 8 | **C** | page Chapter 0; `STORY.md:66-86` | Implies a single clean baseline comparison | Add that the inherited linear-op file is qwen3-a3b-30b-moe data (1,032 rows, ≤4,096 tokens, 20 reps) and that Fig. 1's new curve is job **21309** on the 386-token grid — not the dense job 21313. |
| 9 | **B** | `STORY.md:560`, "Reproducing the figures" | Lists cluster-scratch paths as if available | State that jobs **21356, 21483, 21500 and 21519 exist only under a session-scoped temporary directory outside git**, so four of the eight datasets cannot be reproduced from the repository alone. Record their md5s (see FINDINGS_LOG inventory). |
| 10 | **C** | page Fig. 20 caption | "GEMMs: 1.3–3×" | Consistent with the data, but the caption should say it is a **rolling median over 15 shapes**, i.e. a smoothed curve, and that §0's "1.3–2.9×" refers to the unsmoothed per-TP medians. |

## 2. `evolution/evolution.ipynb`

| # | Sev | Location | Wrong | Corrected |
|---|---|---|---|---|
| 11 | **A** | config cell `DATASETS`; Section 2 `S2_RUNS` | D3 = job 21356 is presented as story step "iter 3" | Job 21356 is the **X1 backlog-knob diagnostic** (4,800 rows, tokens ≥1,024, single-column legacy timer), not a dataset iteration. Label it as a diagnostic, or drop it from the iteration sequence. |
| 12 | **A** | config cell `DATASETS` | Jobs **21500** and **21519** are absent | Both are full 13,308-row dense grids and genuine iterations between 21483 and 21539. Their omission is why the notebook shows 21483 → 21539 as one jump. |
| 13 | **B** | Block 2 markdown, "More tokens cannot mean less time: a bug in the measurement" | States monotonicity as a correctness criterion | "The curves motivated the investigation; the diagnosis rests on the X1 intervention and the trace evidence. Decreases alone prove nothing — the final dataset still has them (see correction 1)." |
| 14 | **C** | Block 4 markdown and `runpos_fig` | "x = timed run index 1..25" | True for the five per-forward scopes. For `emb` the array holds **50 entries (25 forwards × 2 invocations)**, so the x axis is invocation index and the curve alternates between two call sites. |
| 15 | **C** | Inconsistency list item 10 | Already corrected on 2026-10-11 to say the D4 spin-backlog claim comes from STORY.md and is not verifiable from the CSV | No further change; retained here for the record. |

## 3. `presentation_11_10_26/` (my own output of 2026-10-11)

| # | Sev | Location | Wrong | Corrected |
|---|---|---|---|---|
| 16 | **A** | `README.md` (s04 bullet); `manifest.json` s04 caveat | "per-repetition samples are not kept for this dataset" | **False.** Job 21313 stores `time_stats.<op>.samples`: 53 values per shape (3 warm-up + 50 timed), 106 for `emb`. The figure itself is still correct — it plots the stored min/max/median — but the stated reason is wrong, and a true per-repetition figure is possible. |
| 17 | **A** | `README.md` step-mapping table; `make_presentation_figures.py` `STEPS` | "it 2" = job 21356; "it 3" = job 21483; "There is no 'it 4'" | Job 21356 is a diagnostic, not an iteration. The iteration sequence is 21313 → 21334 → 21361 → 21483 → 21500 → 21519 → 21539, so **it 4 = job 21519 does exist**. See `numbers_v2.md` / `manifest_v2.json`. |
| 18 | **A** | `numbers.md`, "Other facts"; `manifest.json` s04 | "max 150–243 ms" | That is TP1 of job 21313 only. Across all TPs: job 21313 **150.4–268.6 ms**, job 21334 **146.4–287.7 ms**. |
| 19 | **B** | `s05_pre_proj_tp8_hostbound_vs_gpubound.png` legend | "it 1: GC removed (host-bound…)" / "it 2: GPU-bound" | Relabel by job: "job 21361 — GC fix, host-bound (legacy timer)" vs "job 21356 — X1 backlog-knob diagnostic, GPU-bound". Also state that the two differ in **clock state as well as queue state** (≈2.0 GHz idle-start vs ≈2.4 GHz after the spin), so the gap is not purely the instrument. |
| 20 | **B** | `s06_rope_fallback_vs_fused.png` | "torch fallback" vs "fused kernel" | Correct, but add that the two curves come from **different jobs with different protocols** (21356: 50 reps, single-column, tokens ≥1,024; 21483: 25 reps, two-column), so the ratio mixes the kernel change with the timing change. |
| 21 | **C** | `appendix_iteration_ratio_to_final.png` | Shows only 21313/21361/21356/21483 against the final | Add jobs **21500 and 21519**; 21519 is the step that shows the medians already at 1.000, which is the appendix's actual question. Done in `appendix_iteration_ratio_to_final_v2.png`. |

## 4. Deck slides (`STORY.pptx` / `build_story_deck.py`) — text and captions only

The user updates the pptx separately. Slide numbers below follow the user's own numbering, which matches the `sNN_`
figure prefixes.

| # | Sev | Slide | Change |
|---|---|---|---|
| 22 | **A** | 2 — "the first data did not look right" | If the slide says the baseline move was from a **gpt-oss** sweep: it was not. The inherited file is qwen3-a3b-30b-moe (20 reps, ≤4,096 tokens). The old page's baseline figure is **attn_pre_proj** against job **21309** (386-token grid), not attn_post_proj and not job 21313. |
| 23 | **A** | 4 — GC spikes | Change "150–290 ms" to a per-job figure: job 21313 **150.4–268.6 ms**, job 21334 **146.4–287.7 ms**. If the slide says "run 11", add "(0-based; the 12th timed run, the 15th forward)". |
| 24 | **B** | 5 — host-time bug | Name the jobs (21361 host-bound vs 21356 GPU-bound diagnostic) rather than "it 1 / it 2", and add the clock-state caveat. The old slide's shaded 4,000–6,000 band is narrower than the data: the host-bound TP8 curve declines from ≈0.094 to ≈0.074 ms across ≈2–6,700 tokens. |
| 25 | **B** | 6 — wrong RoPE kernel | Keep "different computation, not a timing fix". Add that the fallback curve covers only 1,024–2,048 tokens (job 21356's grid) and that the comparison crosses two protocols. |
| 26 | **A** | 7 — before vs after | If the slide repeats "1.3–2.9× / 4–12×", use: GEMM **1.38–2.87×** (pre) and **1.61–2.88×** (post) full-grid, up to **3.25×** / **3.08×** for ≤2,048 tokens; RoPE **4.50–13.02×** full-grid, confounded by the kernel change. Note that 1–11 % of shapes run the other way. |
| 27 | **B** | any slide claiming the dataset is finished | Add that T2 TP4 = 1.9724 fails its pre-registered 2.0 threshold reproducibly, and that the regressor "noise floor" is a cross-collection discrepancy between two jobs, not an instrument constant. |

## 5. Downstream documents that repeat corrected numbers

| # | Sev | Location | Change |
|---|---|---|---|
| 28 | **B** | `regressor_bench/RESULTS_baseline_v2.md:13-15` | Rename "noise floor" to "cross-collection discrepancy (job 21519 settle-3 vs job 21539 settle-8), 1.188 % MAPE". Note the `√2` step is a quadrature argument while `baseline_v2.py:198` subtracts linearly; the two are inconsistent, and `learnable_mape_above_floor` is written to `results.csv` but never published. |
| 29 | **B** | any summary quoting RF "max error 93.9 %" | Add that this is a single extrapolation shape (`num_tokens=1`, the only test token below the train minimum of 2). Excluding it, RF's max error is 17.86 %. |
| 30 | **A** | `operator_boundary_investigation/REPORT.md:190` | "Trace-confirmed five kernels (two copies)" is backed by no trace; E1 marks those internals as assumed, and the real pilot trace shows 3 kernels in decode (5 only in prefill). Rewrite as source-derived. |
| 31 | **B** | `operator_boundary_investigation/REPORT.md:48,396,480` | `execution_time.py:884-907` covers only the MoE all-reduce; the attention one is at `:932-941`. ≈48-line offset through the helper citations. |
| 32 | **B** | `operator_boundary_investigation/REPORT.md:41,50` | "the image has no vLLM" is *inferred* for the approved digest, not observed. "sampler … unmodeled" must keep the "GPU work" qualifier — a CPU-overhead `sampler_e2e_time` exists. |
| 33 | **A** | commit `237dd3e` title / any Dataset B summary | "TP1 eager pilot validated end-to-end" overclaims: the full structural gate returns `structural_pass=false`, 0 of 3,659 kernels reached `valid_context`, and the run was TP1/eager with two requests. |
