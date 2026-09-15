OVERALL: FAIL
BLOCKING_FINDINGS:
  - HIGH: Step 2 / Step 3 — Pilot isolation is broken: Step 3 relies on `WORK_DIR=data/profiling/sweep_work_pilot` and `COLLECT_DIR=data/profiling_pilot`, but Step 2 does not forward `WORK_DIR`, `COLLECT_DIR`, or `LOG_DIR` through `run_in_docker()`. The Dockerized pilot will use the default real sweep/collect trees, so its block-16 cell can make the real block-16 job skip and can contaminate compute outputs — Forward `WORK_DIR`, `COLLECT_DIR`, and `LOG_DIR` in `run_in_docker()` or otherwise give pilot a distinct mounted output/checkpoint path.
  - HIGH: Step 3 — The split fallback is inconsistent with the sbatch contract: the plan says the sbatch reads only `STAGE` and `AITER_BLOCK_SIZES`, while the fallback requires overriding `AITER_TPS` with `"1 2"` / `"4 8"` — Make `AITER_TPS` an explicit overridable sbatch input, e.g. `AITER_TPS="${AITER_TPS:-1 2 4 8}"`, and add the split submission/concat instructions.
NON_BLOCKING_FINDINGS:
  - LOW: Verification — `sbatch --test-only` exercises attention with `AITER_BLOCK_SIZES=16` only; add a block-1 test-only command for symmetry.
CHECKLIST_RESULTS:
  - REQ coverage: FAIL — The plan covers the requested operators, head_dim, pilot, linear job, two AITER cells, and TP=8 rationale, but the pilot output bug can prevent the real block-16 AITER collection.
  - Source traceability: FAIL — Most cited paths and grid counts verify, but the pilot isolation claim contradicts the real sweep script’s Docker env forwarding.
  - Unresolved decisions: FAIL — The TP-split fallback is specified conceptually but not implementable from the stated sbatch inputs.
  - Placeholder scan: PASS — `<sized>` is governed by a pilot sizing rule; no unresolved TODO/TBD placeholder found.
  - Handoff completeness: FAIL — Result ingest and validation are mostly specified, but the pilot/checkpoint contamination path and TP-split concat path leave the handoff unsafe.
FIX_BRIEF:
  - Step 2: forward `WORK_DIR`, `COLLECT_DIR`, and `LOG_DIR` through `run_in_docker()`.
  - Step 3: make `AITER_TPS` an explicit overridable sbatch input and document split-job submission plus Step 4 concatenation.