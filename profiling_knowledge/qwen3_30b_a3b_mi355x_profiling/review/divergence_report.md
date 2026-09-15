# plan-loop divergence report — Qwen3-30B-A3B MI355X profiling plan

Date: 2026-09-10
Plan: profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/01_plan.md
Iteration log: profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/review/iteration_log.yaml
Codex verdicts: profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/review/plan_codex_verdict_iter{1..5}.md

## What happened
Stage 6a (internal) passed every iteration. Stage 6b (Codex, gpt-5.5, read-only) returned FAIL five times.
Every Codex finding was verified against the repository or real data before being applied; none was rejected.
The findings were monotonically narrower: 4 semantic blockers (iter 1) → 4 validator-vs-real-output blockers (iter 2)
→ 3 (iter 3) → 3 internal-consistency residuals (iter 4) → 1 Docker mount/env-forwarding contract (iter 5).
The iteration-5 finding has been applied to the plan text but has not been re-reviewed by Codex because
MAX_REPAIR_ITERATIONS = 5 was reached.

## Why it did not converge
The plan is a long data-collection plan (~600 lines) that binds to many concrete repo behaviours (grid generators,
duplicate keys, replicated-op splitting, true-mixed row semantics, Docker helper). Each Codex pass read a different
part of the source and found one more binding the plan had stated imprecisely. No finding contradicted a user decision.

## Substantive changes made during the loop (user should be aware)
1. `add` residual ops removed from the collected set (fused add+RMSNorm for Qwen3; not emitted by the profiler).
2. Strict `head_dim`: the simulator does NOT validate the dense schema at load; enforcement is profiler write-time +
   `dataset_tools validate` only. This corrects the premise given to the user at the decision gate (grill Q1).
3. Validator recomputes the memory-filtered grid offline from a recorded `gpu_total_bytes` (profiler logs no per-TP count).
4. Duplicate grid keys are by design (2,225×1, 269×2, 3×3 per TP) and are kept; validator asserts that exact multiplicity.
5. aiter prewarm warms all 8 (TP, block) pairs; no dedupe on gqa_ratio (no source authority for AITER's cache key).
6. Linear-op NaN checks scoped per operator/TP; true-mixed `batch_size` semantics honoured.
7. Docker helper must bind-mount `SHARED_ROOT` and forward `WORK_DIR`/`COLLECT_DIR`/`LOG_DIR`; smoke uses its own work namespace.

## Options for the user
- Authorize one more Codex pass (iteration 6) after the grilling answers are folded into the plan; or
- Accept the plan with the iteration-5 fix reviewed internally only (repo policy fallback), documented here.
