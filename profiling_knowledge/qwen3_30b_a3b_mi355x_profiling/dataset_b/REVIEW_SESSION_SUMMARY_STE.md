# Dataset B review session summary

Summary date: 2026-10-21. Last reviewed commit: `25e163c`.

This summary records the session results. It does not include a new review of later code or cluster conditions.

## Goal

Check whether Dataset B can collect reliable operator measurements during actual SGLang execution. Assess whether the code supports this goal without unnecessary complexity.

Dataset B will support a later comparison with isolated operator measurements from Dataset A. This session did not perform that comparison.

## Completed work

- Reviewed the collector, source patch, launch scripts, data pipeline, and acceptance checks.
- Checked the actual job 21798 bundle. Reproduced 3,659 attributed target kernels and 2,150 unresolved kernels.
- Found four problems: incomplete attribution checks, unsafe lock removal, invalid clock context, and lost request phase.
- Wrote the [detailed STE review](TP1_EAGER_REVIEW_STE100.md).
- Reviewed the follow-up fixes in commit `25e163c`.
- Repeated the tests: 151 passed; 1 skipped.
- Confirmed removal of shared-lock deletion and preservation of warmup and measurement phase for target records.
- Confirmed that the original missing-scope test cases now fail correctly.
- Found another false pass when one target scope loses only part of its attribution data.

We reviewed and tested the fixes. We did not implement the collector fixes or start a new GPU collection run.

## Open problems

| Problem | Status at the last review |
| --- | --- |
| Partial attribution loss | Open. A scope with two kernels still passes after removal of one launch record. The unresolved target kernel becomes outside-scope work. |
| Clock interval | Pending live validation. The new pause marker does not, by itself, prove completion of queued GPU work. |
| Sensor resolution | Unknown. Keep the value null until evidence supports a value. Full clock acceptance remains blocked. |
| Model availability | The implementation report states that missing model files blocked job 21837. The user will locate the model. |

The canary also passes without scheduler observations. Its result must not replace the full structural checks.

## Remaining work

1. Correct the partial-attribution false pass.
2. Add a test that removes one launch record from a scope with multiple kernels.
3. Require independent evidence before classification of unresolved kernels as outside-scope work.
4. Repeat a small pilot after the model becomes available.
5. Verify that calibration covers every retained kernel's start and end.
6. Establish sensor resolution or retain the explicit clock acceptance failure.
7. Repeat the acceptance checks and report all remaining failures.
8. Complete the remaining pilot validation before a larger measurement sweep.

The main design is suitable. A complete rewrite is unnecessary. Dataset B acceptance and the Dataset A comparison remain incomplete.
