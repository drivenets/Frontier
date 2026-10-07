# Session handoff — Dataset B (in-situ Qwen3-30B-A3B / SGLang / MI355X kernel timings)

> Current implementation task: [02_IMPLEMENTATION_HANDOFF.md](02_IMPLEMENTATION_HANDOFF.md).
> Image choice and the old universal question gate below are superseded. See the corrected inventory and `IMPLEMENTATION_STATUS.md` for current work. This file retains historical context.

Written 2026-10-04 for whoever picks this up next (human or agent). Everything below is a record of one session's work;
nothing on the cluster has been touched. Read `00_inventory_and_plan.md` in this same directory in full before doing
anything — this file is a map to it and to the surrounding record, not a replacement for it.

## 1. What was asked (verbatim task, lightly reformatted)

The task, pasted into the session as a block titled "Task: build dataset B — in-situ kernel timings for every op our
cost regressors consume, measured inside a real Qwen3-30B-A3B run on MI355X (SGLang)":

- **Why.** Dataset A is Frontier's standalone per-op profiles for `qwen3-a3b-30b-moe` on MI355X (one op at a time, in
  isolation). We don't know whether A reflects what those ops cost inside a real serving run — other kernels before/after,
  weights streamed once per layer, real KV cache, real queue state, real clock behaviour. Dataset B is the reference:
  the same ops, same scope boundaries, measured by GPU-side kernel begin/end timestamps while the full model serves
  real requests. **A-vs-B analysis and fitting are explicitly out of scope** — that is a later phase.
- **Step 0 (mandatory first step):** establish B's scope from the code, not from the requester. Read the profiling code,
  the collection scripts, the trainers/regressors that consume A, and the whole investigation record in
  `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/` (numbered docs, STORY.md, measurement contract, RUN.md files).
  Produce an inventory table of every scope (op) in A that a trainer consumes — code wrapped, shape/key columns, which
  profiling image/version produced it — and present it before proceeding. **That table IS the scope of B**; anything
  not in it is out of scope (don't instrument it, don't extract it).
- **Hard constraints** (learned from A, evidence in the documents): no CUDA/HIP event pairs as the B instrument — use
  kernel timestamps (rocprofv3 primary, torch-profiler/kineto cross-check); no added host work in the hot path beyond
  boundary markers; match A's fixed-collection environment hygiene (env vars/runtime flags, logged per run); clock is
  a first-class measured variable, sampled continuously and joined by timestamp, never locked except as a separate
  optional run; keep every kernel occurrence, one row each, no aggregation at collection time.
- **What to build:** (A) a minimal SGLang patch with range markers around exactly the inventoried scopes, behind an env
  var, confirmed against the real ROCm/AITER execution path, with scope-boundary mismatches reported rather than
  silently approximated; (B) trace extraction to one row per kernel occurrence plus one scope-level row, with a long
  specified column set (run/request/forward ids, phase, layer, scope, kernel name/timestamps, gap-before, clock,
  versions, and A's own key/shape columns per scope); (C) a kineto cross-check pass on at least one config; (D) a
  request driver (concurrency 1, deliberate prompt/generation-length grid straddling A's discontinuities, large chunked
  prefill, radix cache off, decode with and without CUDA graphs, a second pass at larger batch via concurrency); (E)
  configurations at TP=1 and at least one TP>1 that A also covers, using the image we'd actually serve from, versions
  recorded; (F) a checker in the spirit of `sanity_check.py` — every scope present everywhere, stable kernel counts,
  no negative gaps, clock joined everywhere, exact row-count accounting, fail loudly, no vacuous passes; (G) a RUN.md
  per collection, the marker patch as a reviewable diff, and a `B_SCHEMA.md`.
- **Method required:** plan first (inventory table, scope→SGLang mapping, knobs, expected row counts, a written
  prediction of what B looks like if A is context-free) and show it before touching the cluster; then a one-node pilot
  (TP=1, ~10 prompts, both tracers) shown before the full sweep; then the full sweep.
- **Out of scope:** every kernel not in the Step-0 inventory, end-to-end latency/throughput, any A-vs-B comparison or
  fitting, any change to Frontier's profiler or regressors. Any scope-boundary mismatch, image choice, or unrecoverable
  key is a stop-and-ask, not a judgment call.
- **Done when:** B covers every inventory scope across the grid, both tracers on ≥1 config, clock and gap populated,
  checker green, RUN.md + schema written, and a one-page note on everything that could confound an A-vs-B comparison.

## 2. What this session actually did

Only the planning phase (Step 0 + the written plan the task requires before touching the cluster). **No cluster job was
submitted, no SGLang server was started, no GPU was touched.** All investigation was read-only: repo files, the
profiling-knowledge documents, the datasets store on `/opt/shared`, `docker images` listings, and `docker run ... cat`
of source files inside five candidate SGLang images on the Memphis cluster's login node (no container actually ran the
model). Work was parallelized across three background agents (linear_op inventory, attention inventory, MoE +
dataset-catalogue inventory) plus direct investigation of the serving images and cluster state.

Concretely:

1. **Read the full investigation record** behind dataset A: `00_requirements_and_source_map.md`, `01_plan.md`,
   `02_run_record.md`, `05_linear_op_spike_root_cause.md`, `07_post_proj_rope_dip_root_cause.md`,
   `08_measurement_validation_preregistration.md` / `09_…_results.md`, `10_measurement_contract.md`,
   `11_attn_rope_task.md`, `12_dip_investigation_summary.md`, `13_run_position_anomalies_root_cause.md`,
   `14_regressor_dataset_handoff.md`, `HOWTO_run_linear_op_collection.md`, `story/STORY.md` (the full narrative), the
   dataset `README.md` and every `runs/*/RUN.md` under `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/`, and the two
   `sbatch` scripts (`qwen3_mi355x_profiling.sbatch`, `qwen3_fixed_clock_validation.sbatch`).
2. **Dispatched three background agents** to read the actual profiler code and build the inventory:
   - `linear_op` profiler scopes: `frontier/profiling/linear_op/{main.py,linear_op_impl.py,linear_op_wrapper.py}`.
   - `attention` profiler scopes (AITER backend): `frontier/profiling/attention/{main.py,attention_wrapper.py,
     backends/aiter_attention_wrapper.py}`.
   - MoE profiler + a full catalogue of every dataset file/RUN.md for this model on mi355x (confirmed: no usable MoE
     data exists for this model/device — only the legacy "do not use" file).
   Their full reports are in this conversation's transcript (each returned as a structured markdown report); the
   distilled, cited conclusions are folded into §1 of `00_inventory_and_plan.md`.
3. **Inspected five candidate SGLang serving images** on the cluster login node by `docker run --rm --entrypoint bash
   <image> <script>` (reading files, not executing the model): `dnis/sglang-rocm:20260923-2-…` (DriveNets' own build,
   pushed to `10.1.1.0:5050`), `rocm/sgl-dev:v0.5.20-rocm724-…`, `lmsysorg/sglang-rocm:v0.5.19-rocm720-…`,
   `lmsysorg/sglang:v0.5.17-rocm720-mi35x` (A-attention's image), `lmsysorg/sglang:v0.5.11-rocm700-mi35x` (A-linear's
   image). Dumped and read: `srt/models/qwen3_moe.py`, `qwen2_moe.py`, `layers/communicator.py`, `layers/layernorm.py`,
   `layers/rotary_embedding/{base,factory,__init__}.py`, `layers/attention/aiter_backend.py`, `mem_cache/memory_pool.py`,
   `layers/quantization/unquant.py`, `models/utils.py`, `model_executor/model_runner.py`, `environ.py`, `server_args.py`,
   `utils/{nvtx_utils,profile_utils}.py`. This is what produced the scope-mapping table and the fused-RoPE-KV-write
   finding (§4 below).
4. **Checked cluster state**: Slurm queue/partitions, which images are on which nodes, model weights location
   (`/mnt/data/data/hai/hai2/models/Qwen3-30B-A3B`, present on nodes 1,2,4,5,7,8,9; missing on node 6), existing
   DriveNets SGLang launch scripts (`/opt/shared/bp-inference*`, `~/amd-playground/InferenceMAX/sglang_scripts/*.sh`,
   `~/amd-playground/benchmark-pipeline/`), and `rocprofv3`/`roctx` tooling availability inside the recommended image.
5. **Computed A's actual key space and discontinuities** directly from the settle-8 dataset
   (`/opt/shared/.../linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv`, loaded with
   `~/.virtualenvs/qwen3-profiling/bin/python`/pandas) and from the root attention CSVs — the hipBLASLt tile-switch
   token counts, the flat/fixed-cost regions per op/TP, and the decode/prefill shapes — so the grid in the plan is
   derived from measured data, not guessed.
6. **Wrote the Step-0 inventory + full plan**: `00_inventory_and_plan.md` (this directory, ~37 KB, 9 sections). Presented
   it to the user as the required pre-cluster deliverable, with seven explicit stop-and-ask decisions instead of picking
   defaults for anything load-bearing.
7. **Saved a memory note** (outside the repo, user-global): `dataset-b-insitu-plan.md` in the Claude memory project dir,
   indexed in `MEMORY.md`, pointing back at `00_inventory_and_plan.md` and flagging the key findings and the "no cluster
   jobs until Q1–Q7 answered" constraint for future sessions.
8. **Gave the user a plain-language recap** (end of the planning turn) and, on request, an executive summary (goals /
   achieved / open) — both are in the conversation transcript, not duplicated here except as pointers below.

## 3. Where everything lives

| What | Path |
|---|---|
| **The Step-0 inventory + full plan (the actual deliverable)** | `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/00_inventory_and_plan.md` |
| This handoff file | `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/HANDOFF.md` |
| Dataset A's investigation record (read in full) | `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/{00..14}_*.md`, `HOWTO_run_linear_op_collection.md`, `story/STORY.md` |
| Dataset A's data + provenance | `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/README.md` and `runs/*/RUN.md`; large CSVs on `/opt/shared/frontier-qwen3-profiling/datasets/mi355x/qwen3-a3b-30b-moe/{linear_op,attention}/<run>/` |
| The new per-(op,TP) regressor consuming A | `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/regressor_bench/` (`dataset.py`, `README.md`, `baseline_v2.py`) |
| Frontier profiler code read for the inventory | `frontier/profiling/linear_op/{main.py,linear_op_impl.py,linear_op_wrapper.py}`, `frontier/profiling/attention/{main.py,attention_wrapper.py,backends/aiter_attention_wrapper.py}`, `frontier/profiling/moe/` |
| Image inspection scratch output (not committed; session-local) | was under the session's scratchpad dir (`.../scratchpad/images/*.txt`) — **this directory no longer exists** (the environment announced it was removed mid-session); none of it was committed to the repo. If the next agent needs the raw `docker run` dumps again, re-run the inspection (commands are reconstructable from §2.3 of this file and §2–3 of the plan doc). |
| Memory note for future sessions | `~/.claude/projects/-home-dn-Frontier-drivenets/memory/dataset-b-insitu-plan.md`, indexed in that project's `MEMORY.md` |
| Model weights (for the eventual driver) | `/mnt/data/data/hai/hai2/models/Qwen3-30B-A3B` on cluster nodes 1,2,4,5,7,8,9 |
| Candidate serving image (recommended) | `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04`, also at registry `10.1.1.0:5050/dnis/sglang-rocm:20260923-2-...` |

## 4. The headline findings (detail in `00_inventory_and_plan.md` §1–3)

- **A's scope is exactly 9 operators**, split across two sub-datasets collected with *different* measurement regimes:
  - A-linear (6 scopes: `attn_pre_proj`, `attn_rope`, `attn_post_proj`, `input_layernorm`, `post_attention_layernorm`,
    `emb`) — from the settle-8 GPU-bound file, job 21539, image `lmsysorg/sglang:v0.5.11-rocm700-mi35x`.
  - A-attention (3 scopes: `attn_kv_cache_save`, `attn_prefill`, `attn_decode`) — from the **legacy, never-fixed**
    host-bound loop, jobs 21310/21311, image `lmsysorg/sglang:v0.5.17-rocm720-mi35x`. This dataset never received the
    GPU-bound/clock-probe fixes that A-linear did; its small-shape decode/kv-save values are plausibly host launch
    spans, not kernel time. Noted as a confounder, not fixed (out of scope here).
  - The legacy `moe.csv` is explicitly "do not use"; **no MoE data exists for this model on this device** — confirmed
    by direct inspection, not assumed.
- **A scope-boundary mismatch, found by reading the real serving code, not inferred**: on ROCm, this SGLang fuses RoPE
  application and the KV-cache write into a single Triton kernel (`fused_qk_rope_reshape_and_cache`); the gate
  (`enable_fused_set_kv_buffer`) is unconditionally true on HIP. A's `attn_rope` and `attn_kv_cache_save` are two
  separate CUDA-event scopes; in production they are one kernel. This is flagged as the task instructed ("stop and
  ask"), not resolved unilaterally — it's decision Q1 in the plan.
- A's actual grid was computed (not guessed) from the settle-8 CSV: hipBLASLt tile-switch token counts per op/TP (the
  largest a +81% jump at TP4/8208 tokens on `attn_post_proj`), the fixed-cost floor per op/TP, and the attention
  prefill/decode shapes — all written into the plan's grid design (§1.2 and §4.E of the plan doc).
- Five SGLang images were compared on kernel-library versions (sglang, torch/ROCm, aiter, hipBLASLt, rccl, tracer
  availability); none matches either A image, so **image choice is itself a confounder decision**, not a solved
  problem (Q2).

## 5. Open decisions — nothing proceeds until these are answered

These are written out in full, with recommendations, in `00_inventory_and_plan.md` §6. One-line restatement:

1. **Q1 — fused RoPE+KV-write.** Record the fused kernel as one scope (recommended, production path untouched) vs.
   also run an unfused control pass vs. unfused only.
2. **Q2 — serving image.** `dnis/sglang-rocm:20260923-2` (what DriveNets actually serves from; on node 1 only) vs.
   `v0.5.17-rocm720` (matches A-attention's aiter; on 6 nodes).
3. **Q3 — production flags.** `SGLANG_USE_AITER=1` (recommended, otherwise norms/GEMMs silently fall back to
   non-production kernels) and whether `--enable-aiter-allreduce-fusion` is on in real serving (plan recommends off for
   the TP>1 pass so norm and all-reduce stay separable).
4. **Q4 — `DEBUG_CLR_MAX_BATCH_SIZE=1000000`.** Apply A's runtime hygiene fix (recommended, + one control pass without
   it) or leave at the production default.
5. **Q5 — TP>1 choice.** TP4 recommended; TP2/TP8 as optional extra passes.
6. **Q6 — `attn_decode` partition count.** SGLang fixes `max_num_partitions` per server; A computed it per shape. Plan
   proposes recording it as a column and changing nothing — confirm.
7. **Q7 — A-attention as reference.** Confirm B should be compared against A-attention's legacy (host-bound) data as-is,
   given it was never GPU-bound-fixed.

## 6. What is explicitly NOT done yet

- No decisions above have been answered by the user.
- No marker patch has been written (design only, in plan §4.A).
- No trace extractor, checker, or request driver code exists yet (design only, in plan §4.B–D, F).
- No pilot has run (plan §9: one node, TP1, ~10 prompts, both tracers — gated on Q1–Q7).
- No full sweep, no cross-check pass, no locked-clock pass, no `RUN.md`/`B_SCHEMA.md` for B, no confounder note.
- Nothing has been committed to git in this session; the new `dataset_b/` directory is currently untracked
  (`git status` shows it as `??`), as are a few unrelated pre-existing untracked paths in the worktree
  (`data/profiling_gcfix/`, `logs/qwen3_mi355x/`, `regressor_bench_nuzan_sidecar/`, one PNG, one sbatch file) that
  predate this session and were not touched.

## 7. Suggested next step for whoever picks this up

Read `00_inventory_and_plan.md` end to end, then either (a) get the user's answers to Q1–Q7 and proceed to writing the
marker patch + pilot, or (b) if re-verifying the inventory first, re-run the image inspection commands (reconstructable
from plan §2–3) since the prior session's scratch dumps no longer exist on disk — the *conclusions* are preserved in
the plan doc and in §4 above, but the raw `docker run` output files are gone.
