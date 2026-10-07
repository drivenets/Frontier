# Cluster inspection run — approved Dataset B image, 2026-10-04

**No real B measurements exist from this run.** This is runtime/tooling evidence recovery for the
approved serving image only: a CPU-only (`--gres=gpu:0`), 20-minute-capped Slurm job that verified the
exact image digest, ran `inspect_image_b.recover()` inside that image with no GPU/model/server, and
separately checked model/config accessibility on the allocated node. No `sglang_b_markers.patch`
call-site integration, no model serving, no rocprofv3/Kineto trace of a real forward, and no
collection job were run. `sglang_b_markers.patch` still only adds a helper module; it does not
instrument any serving call site.

## Node, job, image

| item | value |
|---|---|
| cluster | Primary (Memphis), partition `XAI` |
| node | `amd-mi355x-3` (idle at job submission; nodes 2/7/9 were running other users' jobs — not touched) |
| Slurm job | `21725` (`smatar-b-inspect`), `--gres=gpu:0`, 20 min cap, CPU-only |
| prior partial job | `21724`, same job name — failed before producing evidence (see "Partial/failed attempt" below); its empty directory is preserved, not deleted |
| registry ref | `10.1.1.0:5050/dnis/sglang-rocm@sha256:d8e0eb9c3211b593fbd574ada06e03d48d956c159bb9b6c123f4d86c12d8e6dc` |
| manifest digest | `sha256:d8e0eb9c3211b593fbd574ada06e03d48d956c159bb9b6c123f4d86c12d8e6dc` (matches the approved value from `operator_boundary_investigation/REPORT.md` / E5) |
| image/config ID (`docker image inspect --format '{{.Id}}'`) | `sha256:dece67677b2b6cfae2aa12a546fab317a1499745d4ec866c45b9bd4ea0fc0bde` — **verified equal** to the expected ID before any evidence was collected (`test "$ACTUAL_ID" = "$EXPECTED_ID"`, job did not proceed otherwise) |
| image presence before this run | absent on `amd-mi355x-3` (consistent with `CLUSTER_ACCESS.md`'s note that it was absent on nodes 1/2/7/8/9 as of the same day) |
| pull | performed by job `21724`'s first (failing) attempt, which ran before the directory-creation bug; the image layers were already local by the time job `21725` ran, so `21725` took the no-pull branch. Disk headroom at pull time: 150G available on `/var/lib/docker`'s filesystem (`/dev/nvme0n1p2`, 98% used, 150G free before pull; 138G free after — ~12G consumed, consistent with the 11.62GB compressed size reported in E5) |

## Partial/failed attempt (job 21724)

The first run failed with `FileExistsError` on `evidence/cluster-inspection-21724-amd-mi355x-3` because
I (the orchestrating session) pre-created and `chmod a+rwx`'d that output directory before invoking
`inspect_image_b.recover()`, which itself calls `output.mkdir(parents=True, exist_ok=False)` and expects
to create it. That pre-creation was my own deviation from the specified script, not a deficiency in
`inspect_image_b.py`. The image pull and image-ID verification inside that same job *did* succeed before
the crash (visible in the job's stdout). The directory is preserved at
`.../dataset_b/evidence/cluster-inspection-21724-amd-mi355x-3/` on the shared checkout; it is empty
(confirmed via `find -ls`) and was not deleted or reused. Job `21725` reran the unmodified script
(no pre-create) and completed.

## Model / config availability

The handoff's historical path, `/mnt/data/data/hai/hai2/models/Qwen3-30B-A3B`, **does not exist** on
`amd-mi355x-3` — `/mnt/data/data/hai/hai2/` exists but only contains `fp8-blockwise-work/`. This is
recorded verbatim in `node_model.txt` ("MODEL CONFIG NOT ACCESSIBLE ON THIS NODE").

Follow-up checks (not part of the original script, done to close the question rather than leave it
unresolved) found `/mnt/data` *is* mounted and populated on this XAI node — contradicting the
cluster-admin skill reference's storage table, which lists `/mnt/data` as ses2-nodes/des2-2 only; that
skill doc looks stale for `amd-mi355x-3` and is worth a separate correction, out of scope here. Under
`/mnt/data/data/hai/` an alternate on-disk copy of the same model was found and its `config.json` read
successfully: `/mnt/data/data/hai/qwen-blockwise-fp8-ab-20260810/models/Qwen3-30B-A3B/config.json`
(sha256 `2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297`; full content in
`model_path_followup.txt`). It describes the expected architecture (`Qwen3MoeForCausalLM`, 48 layers, 32
attention heads, 4 KV heads, head_dim 128, hidden_size 2048, 128 experts/8-per-token, bf16) and is
consistent with the Dataset-A/REPORT.md description of this model. This is a **config-file read only**,
on a sibling directory tree, not a weights/tokenizer completeness check, not a load through
`transformers`/`sglang`, and not proof this is the exact copy future sessions should serve from — flag
it as a candidate path and a sibling `Qwen3-30B-A3B-FP8` / `Qwen3-30B-A3B_torch_dist` exist alongside it,
unopened. Several other `qwen30b`-named work directories under `/mnt/data/data/hai/` (e.g.
`qwen30b-te218-mfu-20260919/*`, with names suggesting both `triton-attn` and `aiter` MI355X serving
smokes for this exact model) were seen in a directory listing only; not opened, not claimed as
production recipes, and not a contradiction asserted against E5's "no Qwen3-30B-A3B recipe found in
`/opt/shared/bp-inferencex`" (different tree, not inspected for recipe content).

## Actual versions (observed inside the running image, this job)

From `versions.json` (`docker run --rm --network none --read-only --tmpfs /tmp --entrypoint python3 <id> -c '...'`):

| package | version |
|---|---|
| sglang | `0.5.18` |
| torch | `2.11.0+gitd0c8b1f` |
| aiter (import name `aiter`, via `importlib.metadata.version('aiter')`) | **not found** — `null` |

The `aiter` lookup returning `null` is a distribution-name mismatch, not evidence the package is
missing: E5 (independent layer inspection) already established the installed wheel is
`amd_aiter-0.1.1.dev1+gb4d9154d1.d20260923` — i.e. the *distribution* is registered as `amd-aiter`
(hyphenated), not `aiter`. `importlib.metadata.version('amd-aiter')` (or `amd_aiter`) would resolve it;
this run did not re-query it under that name, so this run adds no new aiter-version evidence beyond
E5's and does not contradict it. `sglang_root` resolved to
`/usr/local/lib/python3.12/dist-packages/sglang`, confirming the install location used for every
source file this job pulled.

`srt/server_args.py` (10,038 lines, recovered in full) was not previously saved in
`operator_boundary_investigation/evidence/approved_image_source/` (E5 preserved it via a different
earlier channel per the REPORT.md citations, but it is not among the 20 files listed in
`evidence/approved_image_source/identity.json`); this run adds a fresh, image-sourced copy under
`dataset_b/evidence/.../srt__server_args.py`, recovered the same way as the other 19 files (direct
`Path.read_text()` inside the image, no GPU).

## rocprofv3 / ROCTx / toolchain (from `tools.txt`, `docker run ... bash -lc 'rocprofv3 --version; rocprofv3 --help; ldconfig -p | grep roctx; dpkg-query ...'`)

| item | value |
|---|---|
| `rocprofv3` location | `/opt/rocm/bin/rocprofv3` |
| `rocprofv3` version | `1.1.0` (git `c2d94761153e1033a91744842dfc66eddd631fde`) |
| ROCm version (from rocprofv3 `--version`, and confirmed by `dpkg-query`) | `7.2.3` |
| hipBLASLt | `1.2.2.70203-90~22.04` (matches E5 §1's `hipBLASLt 1.2.2` for this image) |
| RCCL | `2.27.7.70203-90~22.04` |
| rocprofiler-sdk / rocprofiler-sdk-roctx | `1.1.0-90~22.04` |
| libroctx | `libroctx64.so.4` and `.so` present at `/opt/rocm-7.2.3/lib/`, plus `librocprofiler-sdk-roctx.so.1`/`.so` — **ROCTx library is present and linkable in this image** |

**Output formats** (`rocprofv3 --help`, "I/O options"): `-f {csv,json,pftrace,otf2,rocpd}` — five
output formats, including the newer `rocpd` (relational/SQLite-style) format alongside the legacy
`csv`/`json`/Perfetto `pftrace`/`otf2`.

**Graph-attribution-relevant flags surfaced by `--help`** (static capability listing only; none of
this was exercised against a real HIP-graph capture/replay in this run):

- `--marker-trace` — "ROCTx Traces... improved ROCTx library with more features" (successor to the old
  `--roctx-trace`).
- `--kernel-rename` — "Use region names defined by roctxRangePush/roctxRangePop regions to rename the
  kernels" (implied by `--kokkos-trace`, available standalone too per the flag's own description).
- `-A {absolute,relative,type-relative}` agent-indexing modes.
- `--selected-regions` — "profile regions of code surrounded by roctxProfilerResume(0)/roctxProfilerPause(0)".
- `rocpd` output format is the most likely candidate for carrying the richer per-dispatch/per-queue
  identity (correlation IDs, agent/queue columns) that graph-node attribution in `operator_boundary_investigation/REPORT.md` §"Attribution and graphs" calls for, but **this run did not inspect the `rocpd` schema** (no trace was captured). That remains the next concrete step for the graph-attribution question (REPORT.md §10.3's proposed one-forward trace), not something this CPU-only, no-model job could resolve.

No claim is made here about whether rocprofv3 1.1.0 can assign repeated kernel names to distinct
graph-replay nodes — `--help` output alone does not demonstrate that; it only shows the flags exist.
Confirming it requires an actual decode-graph capture/replay trace in the pinned toolchain, which
`IMPLEMENTATION_STATUS.md` already lists as an open blocker and this run does not close.

## Source files: recovered vs missing

20 of 22 requested source files (plus `server_args.py` and `http_server.py`, bringing the attempted
total in this job to 22) were recovered with non-empty content, each paired with a `.stderr` file and
a SHA-256 in `sha256.json`. All recovered `.stderr` files are empty (`e3b0c4...` = SHA-256 of the empty
string), i.e. clean reads. Line counts (`wc -l`) confirm non-trivial content for all of them
(570–10,038 lines; see the table embedded in this run's shell transcript / `sha256.json` keys).

Two requested paths **do not exist** in this image and returned empty stdout + a `FileNotFoundError`
traceback on stderr (both logged, nothing silently dropped):

| requested path | status | likely cause |
|---|---|---|
| `srt/model_executor/cuda_graph_runner.py` | missing | superseded by `srt/model_executor/runner/{decode_cuda_graph_runner,eager_runner}.py` — both already confirmed present and DriveNets-modified by `operator_boundary_investigation/evidence/E5...md` §2 (the file was split/moved, consistent with E5's own file list; it did not independently verify this flat-path absence, so this run adds that confirmation) |
| `srt/managers/scheduler_profiler_mixin.py` | missing | not referenced anywhere in E5 or REPORT.md; no replacement path identified in this run — flag as a stale guess in `inspect_image_b.py`'s file list, not a finding about the image's profiling surface |

Neither gap blocks anything in this task: none of the operator-boundary findings in REPORT.md cite
either path.

## What this run does and does not establish

- **Does establish**: the approved image is pullable and present now on `amd-mi355x-3` at the exact
  approved digest/ID; sglang/torch versions inside the running container; rocprofv3 1.1.0 is installed
  with 5 output formats and a linkable ROCTx library; 20 additional exact source files (mostly matching
  E5's prior set, plus `server_args.py`/`http_server.py`) read successfully from the live container
  filesystem, consistent with E5's static wheel-diff findings; the historical model path is absent on
  this node but a plausible alternate copy's `config.json` is readable and matches the expected
  architecture.
- **Does not establish**: any GPU execution, any SGLang server start, any real kernel trace, any
  rocpd-schema graph-attribution proof, any aiter version confirmed under its actual distribution name,
  or model weight/tokenizer completeness. `sglang_b_markers.patch` remains support-only — no call sites
  are patched and no model is instrumented by this evidence-recovery step.

## Paths

- Remote evidence directory (shared NFS checkout, `dn`-owned, world-writable per root-squash
  convention): `/opt/shared/frontier-qwen3-profiling/Frontier/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/evidence/cluster-inspection-21725-amd-mi355x-3/`
- Failed-attempt remote directory (preserved, empty): `/opt/shared/frontier-qwen3-profiling/Frontier/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/evidence/cluster-inspection-21724-amd-mi355x-3/`
- Local copy (this directory): `/home/dn/Frontier-qwen3-profiling/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/evidence/cluster-inspection-21725-amd-mi355x-3/`
- Files in this directory: `commands.json` (exact command list run inside the container), `sha256.json`
  (hash of every output file), `image.txt`/`image.txt.stderr`, `versions.json`, `tools.txt`,
  `node_model.txt`, `model_path_followup.txt` (this session's model-path follow-up, not part of the
  original script), `RUN.md` (this file), and `srt__*.py` / `srt__*.py.stderr` for each requested
  source file.

## Next action (unblocks the still-open items in `IMPLEMENTATION_STATUS.md` §"Remaining blockers")

Do not start model serving yet, per this task's scope. The smallest next steps, in order:
1. Diff the newly recovered `srt__server_args.py`, `srt__entrypoints__http_server.py`,
   `srt__managers__io_struct.py` against upstream `v0.5.18` the same way E5 diffed the other 20 files,
   to extend E5's modified/identical file classification.
2. Resolve the `rocpd` schema question with a real one-forward trace (REPORT.md §10.3) — this is a GPU
   job and is explicitly out of scope for the step this RUN.md documents.
3. Confirm `aiter`'s installed version from inside a container using
   `importlib.metadata.version("amd-aiter")` (or `amd_aiter`) instead of `aiter`, to get a live
   corroboration of E5's static `0.1.1.dev1+gb4d9154d1.d20260923` claim without needing a GPU.
