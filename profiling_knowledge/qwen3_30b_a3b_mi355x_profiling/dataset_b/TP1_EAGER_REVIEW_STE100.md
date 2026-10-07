# Dataset B TP1 eager code review

Review date: 2026-10-07  
Code commit: `237dd3e`  
Review instructions: [REVIEW_INSTRUCTIONS.md](REVIEW_INSTRUCTIONS.md)  
Review decision: Changes required before a larger collection run

## 1 Purpose and decision

Dataset B measures GPU operators while SGLang processes model requests. These measurements will support a later comparison with Dataset A.

The code supports this collection objective. The main design is suitable for the first TP1 eager test. However, the canary check can accept incomplete data.

The review found four defects. Two defects have priority P1. Two defects have priority P2.

| Priority | Meaning |
| --- | --- |
| P1 | Correct the defect before another collection run. |
| P2 | Correct the defect before acceptance of measurement data. |

Keep the main collection design. Correct the acceptance checks and the shared-cache operation. Preserve the distinction between warmup data and measurement data.

## 2 Technical terms

| Term | Meaning in this report |
| --- | --- |
| Dataset A | Measurements of separate operators under controlled test conditions. |
| Dataset B | Measurements of operators during actual model execution. |
| Canary check | A limited check of the first collection result. |
| Target scope | An operator boundary that the scope specification maps to Dataset A. |
| Scope invocation | One execution of a target scope. |
| Kernel dispatch | One GPU kernel execution. |
| Attribution | The connection between a kernel dispatch and its source scope. |
| Forward pass | One execution of the model for a scheduled batch. |
| Clock context | Clock samples and calibration data associated with a kernel dispatch. |
| Drain | The operation that closes collection and saves buffered metadata. |
| TP1 eager | Execution with one tensor-parallel rank and without graph replay. |

Code identifiers retain their original spelling.

## 3 Review scope and evidence

The review covered the collector, source patch, launch scripts, assembly code, extraction code, acceptance checks, and tests.

The local files included the normalized bundle from job 21798. The review used that bundle to check the reported results. No new GPU collection run occurred during the review.

| Check | Result |
| --- | --- |
| Existing test suite | 145 tests passed; 1 test skipped. |
| Target kernels in job 21798 | 3,659 kernels had scope attribution. |
| Extracted target invocations | 2,312 scope invocations. |
| Other kernels | 2,150 kernels had unresolved attribution. |
| Full acceptance check | Stopped at the unresolved attribution error. |
| Canary check | Returned `structural_pass: true`. |

The review also used damaged copies of synthetic test data. These checks tested whether the canary rejects missing records.

| Change to test data | Canary result |
| --- | --- |
| Remove one target marker | Pass |
| Remove one target dispatch and adjust the recorded count | Pass |
| Remove scheduler observations | Pass |
| Set the capture window to an undrained state | Pass |
| Remove telemetry samples | Pass |

The last three results show the limited checks in the canary. They also show why a canary pass cannot replace structural acceptance.

## 4 Finding 1 Missing target data can pass

Priority: P1

Source: [check_dataset_a_scope.py](check_dataset_a_scope.py), lines 27 to 55.  
Related source: [b_extract.py](b_extract.py), function `extract`.

### Condition

The canary gets its expected scopes from `result["scopes"]`. The extractor creates these summaries only for scopes with attributed kernels.

A scope with no attributed kernel has no extracted summary. Thus, it cannot appear in the expected set. The missing-scope check compares two sets that depend on successful attribution.

### Evidence

The canary passed after removal of a target marker. It classified the affected kernel as outside the target set.

The canary also passed after removal of a target dispatch. The original scope metadata still contained the affected invocation.

### Effect

The report can show complete target coverage when target data are missing. An unresolved kernel has insufficient evidence for an outside-scope classification.

### Required actions

1. Calculate expected invocations from forward passes, model layers, and scope applicability rules.
2. Compare these invocations with the original scope metadata.
3. Require at least one attributed kernel for each expected invocation.
4. Keep unresolved attribution separate from confirmed outside-scope work.
5. Use independent evidence before you classify unresolved work as outside scope.
6. Add rejection tests for missing markers, missing invocations, and missing dispatches.

Acceptance criterion: The canary must reject the two damaged target cases described above.

## 5 Finding 2 Shared-cache lock removal is unsafe

Priority: P1

Source: [run_b.sbatch](run_b.sbatch), lines 76 to 86.

### Condition

The script uses `flock` to test a lock file. If the test succeeds, the script removes the file.

A successful `flock` test does not prove that every lock mechanism has released the file. Some build systems use file existence as the lock mechanism.

The test also releases its flock lock before file removal. Another process can acquire the lock during this interval.

### Evidence

A local test kept an exclusive-create lock file open. A separate `flock` test returned success for the same file.

The installed PyTorch `FileBaton` uses exclusive file creation. This review did not verify the exact lock implementation in the cluster AITER installation.

The release-before-removal interval exists independently of that implementation.

### Effect

The script can remove a lock that another build process uses. Concurrent jobs can then write to the same build cache without the intended protection.

### Required actions

1. Remove automatic lock deletion from the shared cache.
2. Use a separate writable build cache for each job when practical.
3. Otherwise, use the recovery procedure for the actual build system.

Acceptance criterion: A collection job must not remove another job's build lock.

## 6 Finding 3 Clock context does not meet acceptance requirements

Priority: P2

Sources: [assemble_bundle.py](assemble_bundle.py), lines 255 to 269; [calibrate_b.py](calibrate_b.py), lines 28 to 30.  
Related sources: [run_b.sbatch](run_b.sbatch), line 57; [telemetry_b.py](telemetry_b.py), class `ClockJoin`.

### Condition

Assembly reserves the final calibration anchors for a separate residual check. The fitted calibration ends at the last fitting anchor.

The current anchors occur at forward-pass entry. Later device work can therefore fall outside the calibration interval.

The launch script also writes a null sensor resolution. The clock check requires a known resolution.

### Evidence

The target kernels in job 21798 have these clock states:

| Clock state | Kernel count |
| --- | ---: |
| `unknown_resolution` | 2,793 |
| `outside_calibration` | 866 |
| `valid_context` | 0 |

The final two measured forward passes account for all 866 kernels outside calibration. Each forward pass contains 433 target kernels.

The saved residual report shows 29 ns for one reserved anchor. This result does not establish valid clock context for the complete capture.

### Effect

The full checker stops at the attribution error before it checks clock quality. Thus, its first error does not identify every acceptance failure.

These clock failures do not, by themselves, invalidate native kernel durations. They prevent acceptance of the required clock context.

### Required actions

1. Collect calibration anchors that bracket the complete device execution window.
2. Select interior anchors for the separate residual check.
3. Establish the sensor resolution from evidence.
4. Preserve an unknown resolution when evidence is unavailable.
5. Report clock quality even when attribution fails.
6. Correct the claim that outside-scope kernels are the only acceptance failure.

Acceptance criterion: Every required target kernel must have valid clock context before full acceptance.

## 7 Finding 4 Assembly removes request phase

Priority: P2

Source: [assemble_bundle.py](assemble_bundle.py), lines 94 to 105.  
Related source: [drive_b.py](drive_b.py), function `send`.

### Condition

The request driver records the warmup or measurement phase. Assembly combines both request logs but does not preserve this field.

### Evidence

The 3,659 target kernels from job 21798 include both phases:

| Request | Phase in the driver log | Target kernels |
| --- | --- | ---: |
| `warmup-0` | Warmup | 1,398 |
| `r-0` | Measurement | 2,261 |

One warmup scope has an elapsed span of approximately 64 seconds. This span is not the sum of its kernel durations.

### Effect

The assembled request records have no explicit phase selection field. Later analysis can include warmup records in measurement statistics by mistake.

Warmup records remain useful for complete request reconciliation. They must retain their phase identity.

### Required actions

1. Preserve the request phase during assembly.
2. Identify measurement records in the extracted output.
3. Keep warmup records available for reconciliation.
4. Exclude warmup records from measurement statistics by default.
5. Test the phase selection with both requests from this pilot.

Acceptance criterion: Measurement selection must not depend on a request-name convention.

## 8 Design assessment

### Parts that support the objective

The source markers identify actual operator boundaries. Launch correlation connects device work to those boundaries. Independent request records support checks of observed execution.

The drain protocol saves metadata before process termination. The extractor keeps kernel duration, kernel-duration sum, and scope span separate. These distinctions support correct measurement analysis.

Keep these parts. Their functions justify their complexity.

### Parts to simplify

Two acceptance paths currently provide different guarantees. The smaller path omits checks that the larger path already contains.

Use one shared structural validator. Give the canary an explicit policy for known outside-scope work. Keep request, window, and coverage checks common.

The runtime also exists inside a maintained patch. Generate that patch from the runtime source to reduce duplicate maintenance.

The evidence directory contains repeated source snapshots and historical explanations. Keep one authoritative source snapshot for each image version. Keep hashes and references for the related evidence.

A complete rewrite is unnecessary. Focus changes on these specific defects and duplicate maintenance tasks.

## 9 Limits of the result

The evidence supports real eager collection and the reported attribution counts. Request reconciliation reaches the attribution check without an earlier failure.

The evidence does not prove complete target coverage or valid clock context. It does not answer the Dataset A comparison question.

Numerical equivalence, instrumentation effects, graph replay, and the remaining pilot matrix remain outside this review's completed validation.

The existing tests provide useful coverage. The additional damaged-data checks show gaps in the new canary tests.

## 10 Completion criteria

1. Correct findings 1 and 2 before another collection run.
2. Correct findings 3 and 4 before measurement acceptance.
3. Repeat the affected synthetic tests.
4. Repeat the canary and full checks on retained pilot evidence.
5. Report every remaining failure separately.
6. Keep Dataset B acceptance false until all applicable requirements pass.

## 11 Language basis

This report uses Simplified Technical English sentence structure and consistent technical terms. The language reference is [ASD-STE100 Issue 9](https://www.asd-ste100.org/).

Full conformity with the controlled dictionary has not been verified.
