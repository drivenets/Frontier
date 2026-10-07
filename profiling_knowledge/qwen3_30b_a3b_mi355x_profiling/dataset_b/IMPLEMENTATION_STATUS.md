# Dataset B implementation status — 2026-10-04

**Superseded by [02_IMPLEMENTATION_HANDOFF.md](02_IMPLEMENTATION_HANDOFF.md)'s successor,
[03_COLLECTION_AGENT_PROMPT.md](03_COLLECTION_AGENT_PROMPT.md), and by real cluster work done against
it.** The paragraph below (SSH/Docker sandbox denial, "no GPU allocation ... was performed") describes an
earlier session and is **no longer current**: SSH/Docker access was resolved (see
`evidence/cluster-inspection-21725-amd-mi355x-3/RUN.md`), and a later session ran real bounded GPU/CPU
jobs on `amd-mi355x-3` — see [evidence/instrumentation_phase_RUN.md](evidence/instrumentation_phase_RUN.md)
for the current state: model/tokenizer verified (real BF16 path, not the stale historical one below), a
real rocprofv3 ROCTX bug found and fixed, and a real source-matched `sglang_b_markers.patch` written,
applied, and import-verified against the approved image on GPU. **Still no SGLang server/pilot run and no
CPU-side scheduler/request metadata hooks** — that remains the next concrete step. Original paragraph,
kept for history:

> Offline collector foundation implemented and tested. No real B measurements, pinned-image inspection, source-matched call-site patch, or pilot were produced. The first milestone remains partial because the workspace denies SSH sockets and Docker daemon access. No GPU allocation, server, commit, push, staging, simulator modification or Dataset A modification was performed.

Current task: [03_COLLECTION_AGENT_PROMPT.md](03_COLLECTION_AGENT_PROMPT.md) (supersedes
`02_IMPLEMENTATION_HANDOFF.md` as the governing task; that file and `00_inventory_and_plan.md` remain
useful history). The old seven-question gate is superseded. The approved image remains
`dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04`; do not substitute or reconfirm it.

## Connection retry and recovered source — 2026-10-04

After reading `CLUSTER_ACCESS.md` and the requested `gpu-cluster:cluster-admin` skill, the SSH smoke test was retried. It still failed before reaching the cluster (`socket: Operation not permitted`). The equivalent Codex outside-sandbox request (`sandbox_permissions=require_escalated`) was explicitly rejected by the tool policy because `sandbox_approval=false`. The Claude-only `dangerouslyDisableSandbox` option is not exposed by this session. User consent is already provided; the remaining access prerequisite is an execution session whose policy permits the network command. Details: `evidence/access_retry_20261004.json`. Local Docker was not retried: the new notes identify this workstation as having no Docker daemon.

The operator investigation's `REPORT.md` and E1–E7 evidence **now exist**. E5 reports the approved registry manifest digest `sha256:d8e0eb9c3211b593fbd574ada06e03d48d956c159bb9b6c123f4d86c12d8e6dc`; this session has not independently contacted the registry. Its downloaded SGLang wheel was found in local scratch and its SHA256 independently matched E5 (`3706017cd830b60144cc964015275257d672f6811e58bcf6e2dfe064b360d75c`). Twenty exact serving source files are now preserved in `evidence/approved_image_source/sglang/`, with provenance and individual hashes in `identity.json`. This removes the missing-source prerequisite for drafting call-site hunks, but does not validate the image runtime or tracer. No operator policy, image substitution, or simulator change was adopted.

## Implemented components

| Files | What works locally | Remaining live dependency |
|---|---|---|
| `b_common.py`, `B_SCHEMA.md`, `scope_spec.json` | identities, actual sequence vectors, padding, model/shard dimensions, consumer aliases, versioned provisional physical mapping | confirm actual CPU metadata sources and per-scope execution sizes |
| `b_traces.py`, `normalize_b.py` | rocprof eager CSV dialect and Kineto complete-event adapters, raw references, nested push/pop pairing, static-label binding | verify pinned native dialect, implement native roctx field adaptation, prove all-rank capture |
| `b_extract.py`, `extract_b.py` | eager launch correlation, same-thread forward containment, ambiguous assignment rejection, pre-filter queue predecessors, per-occurrence export, sum/span/union summaries, reproducible source/input hashes | graph replay attribution deliberately fails; PyArrow is absent, so Parquet roundtrip is untested |
| `b_check.py`, `check_b.py` | observed scheduler/request/rank reconciliation, applicability-based scope counts, explicit exclusions, nonempty data, duplicate/loss/window checks, clock coverage | checks supplied evidence, cannot independently prove asserted manifest flags or unreported interior tracer loss |
| `telemetry_b.py`, `calibrate_b.py` | external sysfs sampling, raw read evidence, GPU-specific calibrated time joins, age/uncertainty/gap states, anchor fitting | resolve GPU sensor paths/update resolution and host/device clock relationship; freeze held-out calibration limits |
| `drive_b.py`, `requests.example.json` | exact input manifests, bounded concurrency, cohorts/offsets, warmup separation, exact response/token/error preservation | verify pinned HTTP response fields, first-token convention and cost of requesting output logprobs |
| `marker_runtime.py`, `sglang_b_markers.patch` | single-backend annotation helper, static label preparation, buffered metadata, overflow/drain guards; patch applies as a new helper module | **support-only patch**: no SGLang call sites are patched; does not instrument a model by itself |
| `inspect_image_b.py`, `run_b.py`, `run_b.sbatch`, `launch_plan.example.json` | pinned-image evidence recovery script and bounded pilot wrapper; refuses unverified configuration; cleans only owned process groups | launch plan intentionally has unresolved ID/commands/verification fields; neither image recovery nor runner was exercised live |
| `audit_a.py`, `workload_grid.json`, `evidence/` | repeatable A inventory and preserved 155-length historical D1 proposal; corrected D2/D3 counts | pilot determines observed forward counts and storage/runtime |

The collector is local to this directory. It has no simulator imports. JSON is the dependency-free interchange; `--parquet` requires PyArrow and fails explicitly if missing. The pilot-sized normalized bundle is loaded in memory; streaming extraction for a multi-million-row sweep is not yet implemented and should be evaluated from pilot size, not assumed safe.

## Validation evidence

- `evidence/pytest.txt`: **135 passed, 1 skipped** after the final code changes. There are 72 concrete positive configurations: TP1/TP4 × prefill/decode/mixed × padded/unpadded × single/overlapping queues × query length 1/8/8192. Additional tests cover chunking, compound scope accounting, repeated names, corruption, missing/duplicate records, graph rejection, clock failure modes, calibration, exact consumer aliases, driver errors, buffered markers and launch refusal. These are synthetic collector tests, not 72 model executions.
- `evidence/synthetic_validation/bundle.json`: invented TP4/mixed/padded/overlap records. `extracted-current/check.json`: structural PASS, `dataset_accepted=false`; 216 raw kernels = 208 targets + 8 exclusions; 104 physical scopes; 8 rank-local forwards = 2 global forwards × 4 ranks; 2 reconciled requests. `extracted/` retains the earlier local extraction; use `extracted-current/` for the source-hashed result.
- `evidence/a_inventory.json`: A-linear MD5 `3635e4d305ae5cf84884e275d3ff37ef`, 13,308 × 150; all six labels/TP coverage audited; TP1/TP4 post-projection transitions +30.07%/+80.67%. Attention 25,032 = 22,152 standard + 2,880 mixed; all mixed decode features and three labels populated. Dataset files were read only.
- `evidence/native_dialect_reference.json`: headers/checksums of unrelated existing rocprof integration CSVs. Parsing correctly rejected a corrupted timestamp containing HIP API text. This is neither pinned-image validation nor B data; no salvage or substitution was performed.
- `evidence/local_checks.json`: shell syntax, Python compilation, helper patch dry-run/application, example manifest validation, expected preflight refusal, and expected full-dataset acceptance refusal.
- Parquet roundtrip is the **one skipped test** because `pyarrow` is not installed. The missing-dependency failure path is tested; no Parquet files are claimed.

Repository gates were attempted without modifying unrelated code. The prescribed PDD doc/script tests yielded **10 passed, 3 failed** on assertions about untouched `README.md`. The two AGENTS-listed `tests/debug/e2e-level/monolith_mode/scripts/*dummy.sh` paths do not exist here. The current dense co-location example was tried instead and failed before simulation with `ModuleNotFoundError: plotly`; log: `evidence/dense_simulator_smoke.log`. The full repository gate of 50 diverse dense/MoE, online/offline simulator scenarios is **unmet**. The collector's 72 synthetic combinations are not presented as satisfying that gate. No simulator numeric/fidelity baseline was run.

## Reproduce local work

From the repository root:

```bash
B=profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b
PY=/home/dn/.virtualenvs/qwen3-profiling/bin/python
$PY -m pytest "$B/tests" -q
$PY "$B/check_b.py" "$B/evidence/synthetic_validation/bundle.json"
$PY "$B/extract_b.py" "$B/evidence/synthetic_validation/bundle.json" /tmp/b-extraction-new
$PY "$B/drive_b.py" "$B/requests.example.json" --validate-only --output /tmp/b-unused
$PY "$B/audit_a.py" --output /tmp/b-audit-new.json
```

All output paths must be new. To request Parquet in an environment with PyArrow, add `--parquet` to extraction. `check_b.py --require-dataset-accepted` intentionally exits nonzero even for the structurally valid example. b0.1 cannot certify a full dataset.

For eventual real traces, independently collected metadata supplies `process_sources` with per-process native paths/maps, scheduler/request records, paired ranges or normalized marker events, telemetry and manifest. Run:

```bash
$PY "$B/normalize_b.py" /path/to/metadata.json /path/to/bundle-new.json
$PY "$B/extract_b.py" /path/to/bundle-new.json /path/to/extracted-new --parquet
$PY "$B/check_b.py" /path/to/bundle-new.json --report /path/to/check-new.json
```

See `B_SCHEMA.md` and the saved synthetic bundle for the exact contract. Do not fabricate paired ranges, scope metadata, calibration or diagnostics to adapt a native trace that lacks them.

## Remaining blockers and smallest next actions

1. **Environment access.** `evidence/access.txt` records SSH `socket: Operation not permitted` and Docker socket denial. No approval escalation is available in this workspace. In an environment already permitted to use `ssh cluster`, copy only this task directory to the shared checkout, then run `python3 inspect_image_b.py <new-evidence-directory>` there. This reads the approved image without model/GPU execution; it records commands, immutable local ID/repo digests, source files/hashes, versions and profiler help. Verify model/config access on the intended node, package location, server CLI, profiling controls and output permissions before allocation.
2. **Real call-site integration.** Read recovered sources, apply the support patch to an extracted source tree (`patch -p1 < sglang_b_markers.patch`; source-layout path is `python/sglang/srt/frontier_b_markers.py`), then add reviewable exact call-site hunks. Install the helper at the image's inspected package location. Scope only QKV+QK norm, output GEMM, actual norm calls, embedding and source-verified attention/RoPE/cache boundaries; communicator wrappers include excluded work. Bind forward IDs across scheduler/ranks, avoid tensor-to-host reads, flush only after all workers/threads and traces drain. Preserve source and final patch hashes. Do not mark `callsite_patch_verified` true for the present helper-only patch.
3. **Operator policy.** `../operator_boundary_investigation/REPORT.md` was absent at the original implementation check and is now available (see retry update above). No investigation files were modified. Incorporate its source findings; resolve material fused RoPE/cache, mixed-attention, TP norm/communication and embedding consequences with concrete source evidence. `qwen3-provisional-0.1` is a structural test recipe, not approved final modeling policy. Unambiguous exploratory eager scopes can proceed; full coverage cannot be frozen.
4. **Native tracer / graphs / clocks.** Verify the pinned roctx format and all-rank dispatch/correlation identity with the smallest trace. Keep richer JSON/other native formats if needed for capture/node graph identity. Do not match 48 layers by repeated names or upgrade the image. Calibrate per-GPU sensor resolution/timebase; the fixture limits are invented. If replay remains unresolved, report it while validating eager data.
5. **Pilot validation.** Fill a verified launch plan with immutable image ID, effective BF16/page-1/radix-off/automatic-clock settings, source hashes, model path, actual rocprof/Kineto and sampler command arrays, explicit profiling/drain controls, and unique paths. `run_b.py PLAN --preflight-only` must pass before `sbatch`. Then use `run_b.sbatch` with its required `B_*` variables under XAI; it is capped at 30 minutes. Start TP1, then matched Kineto and all-rank TP4. Test short/long/chunked/mixed/padded cases, marked/unmarked outputs and perturbation; preserve failed attempts. No full sweep until pilot results are presented and mapping is resolved.
6. **Local dependency/gates.** Make PyArrow available to exercise compressed export, and use a simulator environment containing its declared dependencies for the still-unmet broad regression gate. Keep the unrelated README test failures visible; do not change them as part of this collector task.

## Comparison confounders retained

A-linear and A-attention use different images and measurement regimes. B's approved image changes kernel libraries, fusion and real model/KV/cache activity. A-linear has event overhead/backlog/settling history; A-attention has possible host contamination whose per-row decomposition is not measured. Layer-0 norm, TP embedding, compound RoPE/cache and server-wide decode partition counts may change scope comparability. `DEBUG_CLR_MAX_BATCH_SIZE=1000000` is the planned main setting and a possible departure from ordinary production. Synthetic token inputs and requesting output token logprobs can affect serving activity; measure their influence. Clock samples provide context, not instantaneous per-kernel frequency. These caveats do not invalidate collection, but prevent attributing a later A–B difference solely to execution context. No A-versus-B comparison or regressor fitting was performed.
