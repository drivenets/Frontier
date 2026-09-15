OVERALL: FAIL
BLOCKING_FINDINGS:
  - BLOCKER: implement-loop Handoff / AC-2 — AC-2 says `aiter_prewarm` derives 4 templates for Qwen3, but D-2/TASK-2 require TP {1,2,4,8} × block_size {1,16} = 8 specs/warm calls — Change AC-2 to require 8 Qwen3 specs/warm calls and preserved gpt-oss shard shapes for every requested TP × block_size pair.
  - BLOCKER: D-4 / D-7 / Verification Commands — The Qwen driver env list does not set `LOG_DIR`; the reused sweep defaults logs to `$WORK_DIR/logs`, while ingest/validate expect `logs/qwen3_mi355x` and `smoke.log`, so cell-log checks and `gpu_total_bytes` handoff are broken — Add `LOG_DIR=/opt/shared/frontier-qwen3-profiling/logs` to the driver/sweep contract and specify creation/ingest of `smoke.log`, or update ingest/validate commands to consume `$WORK_DIR/logs` and the actual Slurm output path.
  - BLOCKER: Verification Commands — The shell syntax loop uses `bash -n "$f" || echo ...`, which can still exit 0 after a syntax failure if a later file passes — Make the loop fail nonzero on any syntax error, e.g. accumulate `failed=1` and `exit "$failed"`.
NON_BLOCKING_FINDINGS:
  - LOW: D-3 / dataset_tools — `expected_attention_grid` should explicitly carry or hard-code `max_pipeline_parallel_size=1` in the expectation contract so the offline memory-filter recomputation cannot drift from the profiling command.
CHECKLIST_RESULTS:
  - REQ coverage: FAIL — Core profiling requirements are covered, but REQ-12/REQ-13 log and provenance validation handoff is incomplete.
  - Source traceability: FAIL — The 4-template AC-2 statement contradicts the verified TP × block-size source-derived grid.
  - Unresolved decisions: PASS — Branch source, strict legacy handling, MoE deferral, and canonical AITER files are recorded.
  - Placeholder scan: PASS — Implementation placeholders are explicit and bounded.
  - Handoff completeness: FAIL — AC-2, log paths, `smoke.log`, and the syntax-check command need correction before implement-loop.
FIX_BRIEF:
  - implement-loop Handoff / AC-2: replace “derives 4 templates for Qwen3” with “derives 8 Qwen3 specs/warm calls: TP {1,2,4,8} × block_size {1,16}.”
  - D-4 / script signature: add `LOG_DIR=/opt/shared/frontier-qwen3-profiling/logs` and require the smoke stage or ingest step to produce `logs/qwen3_mi355x/smoke.log`.
  - Verification Commands: replace the `bash -n` loop with a nonzero-failing loop.