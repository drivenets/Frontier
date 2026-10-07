# Dataset B — Jira ticket creation agent prompt

Create Jira tickets for the Dataset B collection work, following my established Jira conventions: project, issue types, hierarchy, naming, labels, ownership, and description format.

Use the actual implementation state to distinguish completed work from remaining work. Read:

Repository:

```text
/home/dn/Frontier-qwen3-profiling
```

Task directory:

```text
profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/
```

Key documents:

- [03_COLLECTION_AGENT_PROMPT.md](03_COLLECTION_AGENT_PROMPT.md) — complete task and acceptance requirements
- [02_IMPLEMENTATION_HANDOFF.md](02_IMPLEMENTATION_HANDOFF.md) — measurement constraints and scope
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — earlier implementation state
- [Instrumentation phase RUN.md](evidence/instrumentation_phase_RUN.md) — subsequent instrumentation progress
- [B_SCHEMA.md](B_SCHEMA.md)
- [workload_grid.json](workload_grid.json)
- [Cluster inspection RUN.md](evidence/cluster-inspection-21725-amd-mi355x-3/RUN.md)
- [Operator investigation REPORT.md](../operator_boundary_investigation/REPORT.md)

Check for newer run records before creating tickets. Where documents conflict, prefer the latest concrete execution evidence and explicit user decisions.

## Objective

Collect reproducible, occurrence-level GPU kernel measurements inside complete Qwen3-30B-A3B serving runs on MI355X using the approved SGLang image. Preserve actual shapes, scope attribution, request/forward/rank/device identities, telemetry, raw traces, and provenance for later comparison with Dataset A.

A-versus-B analysis, regressor fitting, simulator changes, and Dataset A recollection are outside this task.

## Current known state — verify before using

Completed:

- Approved image recovered and verified on the cluster.
- Candidate BF16 model and tokenizer checked.
- Offline collector/schema/checker foundation and synthetic tests.
- Correct ROCTX library integration and native marker CSV adapter.
- Eager correlation mechanism demonstrated with real GPU GEMM kernels.
- Actual serving call-site patch applied and import-tested in the approved image.

Not yet demonstrated in that record:

- Scheduler/request/forward metadata hooks.
- A real SGLang request through the complete collector.
- Calibrated per-GPU telemetry.
- Graph replay attribution.
- All-rank TP4 validation.
- Full pilot matrix, numerical equivalence, or perturbation checks.
- Full Dataset B sweep.

Do not describe the GEMM probe or import test as a completed serving pilot or collected B data.

## Ticket structure

Use an epic or parent issue if that matches my conventions. Create independently actionable child tickets around these work packages, merging or splitting where appropriate:

1. **Complete serving metadata and collector integration.** Scheduler/request membership, actual sequence vectors, padding, cross-rank forward correspondence, process/GPU identities, buffered metadata and verified drain, effective launch configuration.
2. **Validate the first end-to-end TP1 eager collection.** Serve real requests; normalize, extract, and check actual traces; verify physical scope placement and raw-to-extracted accounting; preserve real regression fixtures.
3. **Implement and validate telemetry/timebase calibration.** Continuous per-GPU sampling, sensor resolution, read latency, calibration, freshness/uncertainty limits, missing/stale/wrong-device rejection.
4. **Demonstrate graph replay attribution.** Capture-derived or native graph/node evidence on the pinned toolchain, repeated layers, replay identity, and real versus padded shapes. Kernel-name/order guessing is unacceptable.
5. **Complete the pilot and measurement validation.** Short/long prefill, decode, chunking, actual mixed execution, padding, every TP4 rank, matched Kineto cases, marked/unmarked output equivalence, and marker/tracer perturbation.
6. **Freeze coverage and prepare the sweep.** Versioned physical scopes and A correspondence, executable D1/D2/D3 plus mixed/padding manifests, reproducible workload ordering, bounded jobs, runtime/storage estimates, and scalable extraction.
7. **Collect and validate the full Dataset B sweep.** TP1/TP4 and required eager/graph regimes; retain all occurrences and ranks; reconcile actual execution with requested coverage; preserve failures and recollect invalid/missing cells explicitly.
8. **Package and hand off Dataset B.** Immutable raw traces, telemetry, occurrence-level Parquet, derived tables, manifests/hashes, source patch, commands, checker reports, `RUN.md` files, coverage matrix, and confounder note.

Treat completed infrastructure/instrumentation work as linked context or completed issues only if that matches my conventions and is supported by evidence. Do not turn each script or individual test case into a separate ticket.

## Dependencies and acceptance

Make dependencies explicit. Independent implementation can proceed in parallel. The full sweep depends on a validated pilot, resolved material scope decisions, and a frozen coverage manifest.

Each ticket should contain:

- Concrete outcome and scope.
- Current state and remaining work.
- Verifiable acceptance criteria.
- Dependencies/blockers.
- Relevant repository and artifact references.
- Expected deliverables.

Carry these requirements into the relevant acceptance criteria:

- GPU kernel timestamps; no added HIP/CUDA event timing, backlog kernels, settling spins, or main-run clock locks.
- One physical duration for fused work; no guessed splits or double counting.
- Every participating rank and request reconciled against independent execution metadata.
- Actual sequence-length vectors and padding; HTTP concurrency is not observed batch size.
- Separate kernel duration, scope kernel sum, and scope span.
- Nonempty results, unique attribution, explicit exclusions, trace/window completeness, and valid clock coverage.
- Synthetic tests alone cannot establish GPU correctness.
- Present pilot results before the full sweep.
- No arbitrary universal tracer-agreement threshold.
- Preserve failed attempts and partial coverage honestly.

## Execution

Search Jira first for existing Dataset B or related profiling issues. Update/link existing issues where appropriate; avoid duplicates. Then create the necessary issues and dependency links using my conventions.

Do not invent owners, estimates, deadlines, priorities, or completed statuses unsupported by my conventions or the evidence. Ask only for required information you cannot infer.

Return the created/updated issue keys and links, the hierarchy and dependencies, and any remaining decisions.
