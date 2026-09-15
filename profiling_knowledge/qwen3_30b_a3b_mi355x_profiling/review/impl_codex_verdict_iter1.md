OVERALL: PASS
BLOCKING_FINDINGS:
  - none
NON_BLOCKING_FINDINGS:
  - none
CHECKLIST_RESULTS:
  - Acceptance criteria met: PASS — `head_dim` is in dense required profiling columns and all three attention row dicts; CUDA timing keeps warm-ups plus active samples with timed-only aggregates; wrappers use `ACTIVE_STEPS = 50`.
  - Tests pin the behaviour (RED/GREEN evidence, regression for the blocker): PASS — schema tuple/rejection tests and TimerStatsStore per-scope warm-up tests cover the correctness-sensitive paths; I did not rerun tests due read-only/no-write instruction.
  - No regression for other TimerStatsStore callers (moe_wrapper, collectives_wrapper) or the record_function branches: PASS — MoE/collectives never call `mark_warmup_end`, so `warmup_count` remains 0 and aggregates stay over recorded active samples; record_function paths do not use `TimerStatsStore.get_stats()`.
  - Scope discipline (only Steps 1 and 1b; minimal-code policy; no new abstractions): PASS — changes are limited to the requested schema, profiling wrappers, stats store, and focused tests.
FIX_BRIEF:
  - none