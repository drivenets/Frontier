# Plan: Qwen3-30B-A3B MI355X operator profiling — AITER attention + attention-layer linear ops

## Goal

Produce a verified implementation plan for collecting fresh MI355X profiling data (AITER backend, BF16, cuda_event) for a selected set of Qwen3-30B-A3B (`qwen3-a3b-30b-moe`) operators, with the head_dim schema fix, such that the resulting CSVs pass a written validation checklist and are ready for regressor training in the next phase.

## User Request Summary

From `/home/dn/qwen_profiling_plan_01.md` (2026-09-10): Repo Frontier-drivenets, model Qwen3-30B-A3B (`qwen3-a3b-30b-moe`), new branch `smatar/qwen3-30b-mi355-profiling`. Frontier simulates operators; per-operator execution time comes from regressors trained on profiler CSVs under `data/profiling/compute/<device>/<model>/`. This model is already architecturally faithful in Frontier, so the workstream is pure profiling / data engineering. Task: (1) select several operators from the model's real Frontier operator manifest to build regressors for; (2) collect fresh profiling data for them on MI355X. Out of scope: regressor fitting, any simulator/binding/precision code change, any other model. Step one: enumerate the real manifest from `frontier/operators/binding.py` and `frontier/profiling/`. Selection criteria: (a) TTFT/TPOT share, (b) exposure to extrapolation, (c) portable reference kernel vs production kernel path. Environment: MI355X, AITER backend (sglang's kernels), reuse the gpt-oss sweep scaffolding; block_size ∈ {1, 16} (exact-match filter); keep the output file set consistent with `attention_dataset_contract.py`; add `head_dim` to the recorded + required dense columns; sweep TP ∈ {1,2,4,8}, contexts ≥ 16k, true-mixed on, chunked-prefill grid on. Deliverables: (1) operator list with reasoning, (2) exact grid per operator, (3) new vs reused scripts/config, (4) output locations + schema incl. head_dim, (5) validation checklist (row counts, grid coverage, schema/no legacy drift, no duplicate/symlinked data).

Follow-up instruction (this session): work in a new worktree, create a task folder holding the plan, write a detailed plan.

## Context

- Frontier resolves an operator manifest per model config (`frontier/operators/binding.py:52-110`). For `qwen3-a3b-30b-moe` the manifest resolved on this checkout is `dense_attention/gqa` + `memory/replicated` + `moe/routed`; no shared expert, no dense FFN (record in `00_requirements_and_source_map.md`, "Stage-one enumeration").
- The dense attention family's schema is `DENSE_ATTENTION_FAMILY.required_profiling_feature_columns` (`frontier/attention/families.py:79-91`). It has no `head_dim`; the predictor's exact-match filter uses `n_embd`, `n_q_head`, `n_kv_head`, `block_size`, TP (`sklearn_execution_time_predictor.py:1222-1228`). The profiling `ModelConfig` already carries the explicit `head_dim=128` for this model (`frontier/profiling/common/model_config.py:455-463`, `data/config/models/qwen3-a3b-30b-moe.json`).
- The AITER dense/GQA wrapper (`frontier/profiling/attention/backends/aiter_attention_wrapper.py`) and the gpt-oss sweep template (`profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh`) are on `feat/gptoss120b-mi355`; per the user's Stage 3 decision this branch is based on that commit (c68096c), so both are present in the worktree.
- The MoE profiler's grouped GEMM path is vLLM's Triton `fused_moe` only (`frontier/profiling/moe/moe_wrapper.py:540-580`); sglang production uses `--moe-runner-backend aiter` (`profiling_knowledge/AITER_KERNELS.md`). The user decided to defer MoE collection entirely.
- Execution environment (probed 2026-09-10): Slurm partition `XAI`, nodes `amd-mi355x-[1-9]` (gfx950, 8 GPUs); `/opt/shared` is NFS shared across nodes, `/data` and `/home/dn` are node-local; image `lmsysorg/sglang:v0.5.11-rocm700-mi35x` on the nodes contains torch 2.9.0a0, vllm, sglang, aiter and `/opt/rocm/bin/hipconfig`; `aiter.ops.attention.paged_attention_ragged` and `aiter.ops.mha.mha_batch_prefill_func` signatures match the wrapper's calls; `/mnt/data/aiter-cache` exists per node; `/dev/kfd` gid 110. GPU jobs run via `ssh cluster "sudo -u dn sbatch …"` (project memory rule).
- Existing MI355X Qwen3 attention data is TORCH_SDPA, block 16, `max_model_len` 9472 (context only; superseded).

## Reliable Sources

### Authoritative (implementation truth)

| Rank | Name | Location | Covers |
|------|------|----------|--------|
| 1 | Request brief | `/home/dn/qwen_profiling_plan_01.md` | goal, scope, constraints, sweep axes, deliverables |
| 2 | Operator binding | `frontier/operators/binding.py:52-110`; `frontier/operators/families.py` | manifest resolution |
| 2 | Dense attention family spec | `frontier/attention/families.py:51-98` | dense schema; head_dim insertion point |
| 2 | Attention profiler entry point | `frontier/profiling/attention/main.py:361-666` (args), `870-892` (write-time validation), `1760-1860` (memory filter), `1912-2020` (outputs) | CLI, outputs, config yaml |
| 2 | Attention wrapper row builders | `frontier/profiling/attention/attention_wrapper.py:388-402, 493-503, 563-573` | where head_dim is recorded |
| 2 | Profiling ModelConfig | `frontier/profiling/common/model_config.py:286, 422-454, 455-463, 557` | head_dim source; KV-head replication at TP=8 |
| 2 | Grid generators | `frontier/profiling/utils/__init__.py:156-218, 242-411, 885-934` | point counts (2772 standard / 360 true-mixed per TP at 16384; 387 num_tokens at 16384) |
| 2 | AITER dense wrapper | `frontier/profiling/attention/backends/aiter_attention_wrapper.py` (commit 645c98a) | kernel calls, caveats |
| 2 | gpt-oss sweep template | `profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh` | per-cell split, prewarm, docker, collect |
| 2 | Attention dataset contract | `frontier/execution_time_predictor/attention_dataset_contract.py` | sibling-file rule |
| 2 | Predictor attention loader | `frontier/execution_time_predictor/sklearn_execution_time_predictor.py:1195-1238`; `shared_prediction_model_manager.py:1672-1690` | dense exact-match filter (no schema validation); MLA-only load-time validation |
| 2 | Attention TP policy | `frontier/execution_time_predictor/attention_tp_policy.py:26-32, 54-108` | TP=8 valid with 4 KV heads; attention linear ops set |
| 2 | linear_op profiler | `frontier/profiling/linear_op/main.py:224-400`; `linear_op_impl.py:425-528` | CLI, `--is_moe`, QK-norm inside `attn_pre_proj` |
| 2 | MoE profiler | `frontier/profiling/moe/main.py:166-360`; `moe_wrapper.py:540-580` | grouped GEMM backend = vLLM fused_moe |
| 2 | MI355X collection driver | `examples/profiling/profile_mi355x.sh` | linear_op invocation, num_tokens=4000 exclusion, rope fallback |
| 2 | Model config | `data/config/models/qwen3-a3b-30b-moe.json` | 32/4 heads, head_dim 128, hidden 2048, 48 layers, bf16 |
| 2 | Schema-pinning tests | `tests/unit/test_attention_family_specs.py:765-777`; `test_attention_family_spec_data.py:25-37`; `test_attention_main_true_mixed_contract.py:130-215`; `test_attention_tp_effective_mapping.py:83-157`; `test_execution_time_predictor_max_tokens_budget.py:122-135, 345-470`; `test_shared_prediction_model_manager_eager_attention_decode.py:63-139`; `test_spec_decode_predictor_verify_prefill_path.py:186-290`; `test_mha_stage2_profile_modeling.py`; `test_mqa_stage2_profile_modeling.py` | tests that must change with head_dim |
| 2 | Cluster environment probe (2026-09-10) | `ssh cluster` (= amd-mi355x-1): `sinfo`, `srun -p XAI`, `docker run … python -c "import aiter, vllm; inspect.signature(...)"` | partition, image, shared FS, kernel signatures |
| 2 | Repo development gates | `AGENTS.md` "Development Gates" | no duplication, preserve behaviour |

### Context-Only (not implementation truth)

| Rank | Name | Location | Note |
|------|------|----------|------|
| 3 | GPTOSS_TRUE_MIXED_BATCH_PROFILING.md | `profiling_knowledge/` | 86-hour lesson, block_size facts, sanity checks |
| 3 | AITER_KERNELS.md, MI355X_ROCM_COOKBOOK.md, INFRASTRUCTURE_MAP.md, MI355X_FOUR_MODEL_PROFILING.md | `profiling_knowledge/` | gotchas; partially stale |
| 3 | Frontier fidelity audit | `amd-playground/poc-mi355x-qkv-gemm-profile/frontier_fidelity/01_frontier_architecture_fidelity_audit.md` | why this model; head_dim gap |
| 3 | Existing MI355X Qwen3 CSVs | `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/*.csv` | superseded TORCH_SDPA data |
| 3 | Project memory | `~/.claude/projects/-home-dn-amd-playground/memory/project_slurm_workflow.md`, `feedback_slurm_sudo.md`, `feedback_forgeloop_workarounds.md` | sbatch + `sudo -u dn` rule; Codex `exec` form |

## Requirements

- REQ-1: Enumerate the operator manifest Frontier actually resolves for `qwen3-a3b-30b-moe` and derive the operator list from it. — Request "STEP ONE"; `binding.py`.
- REQ-2: Select "several" operators using criteria (a) TTFT/TPOT share, (b) extrapolation exposure, (c) portable vs production kernel; give include/exclude reasoning for every manifest operator. — Request "SELECTION CRITERIA", deliverable 1.
- REQ-3: Attention is profiled with the AITER backend (production kernels), not TORCH_SDPA. — Request "TARGET ENVIRONMENT".
- REQ-4: Attention profiled explicitly at `block_size ∈ {1, 16}`. — Request.
- REQ-5: Sweep axes: TP ∈ {1,2,4,8}; `max_seq_len = max_model_len ≥ 16384`; true-mixed batches on; chunked-prefill grid search on. — Request.
- REQ-6: `head_dim` is recorded in every dense attention row (standard, mixed, true-mixed) and is a required dense-family column. Enforcement points are the profiler write path (`main.py:886`, `_prepare_standard_attention_output_dataframe`) and `dataset_tools validate`; the predictor load path validates only the MLA family (`sklearn_execution_time_predictor.py:1233-1238`, `shared_prediction_model_manager.py:1682-1686`), so legacy dense CSVs are rejected by those two gates only and keep loading in the simulator. No migration, no new load-time gate (simulator code is a non-goal). — Request; user decision D2, premise corrected in plan-loop repair iteration 1.
- REQ-7: Output file set is safe under `attention_dataset_contract.py` and follows the exact-match `block_size` rule: canonical trio = union of AITER block-1 and block-16 rows; per-cell `_aiter_block{N}` files kept; legacy TORCH_SDPA trio renamed `*_sdpa_block16.csv`. — Request; user decision D4.
- REQ-8: Reuse the gpt-oss sweep scaffolding instead of new infra; changes to it must preserve gpt-oss behaviour. — Request; `AGENTS.md` gates.
- REQ-9: Attention-layer linear ops (`attn_pre_proj` incl. QK-norm, `attn_post_proj`, `attn_rope`) plus the replicated memory ops the profiler emits for this model (`input_layernorm`, `post_attention_layernorm`, `emb`; `add` is fused into layernorm) are re-collected with `num_tokens` extended to 16384 so linear regressors are not extrapolating at 16k prefill chunks. — REQ-2 criterion (b); `linear_op` profiler.
- REQ-10: MoE operators are excluded from this plan's collection and the reason (no AITER grouped-GEMM path) is recorded. — User decision D3.
- REQ-11: All GPU work runs on the Slurm cluster (`XAI`, `sudo -u dn sbatch`) inside `lmsysorg/sglang:v0.5.11-rocm700-mi35x`, with code/results on `/opt/shared`. — Project memory rule; probe.
- REQ-12: A validation checklist exists as an executable check (schema incl. head_dim, value sets, grid coverage, row counts, no NaN/zero medians, combined = standard + true-mixed, no duplicates, no symlinks, contract passes) and as prose. — Request deliverable 5.
- REQ-13: A run manifest records image, commit, node, wall time, row counts, checksums for every collected file. — Request deliverable 4 (provenance); INFRASTRUCTURE_MAP "verify with a checksum".
- REQ-14: The branch is based on `feat/gptoss120b-mi355` (c68096c) rather than `main`. — User decision D1.

## Non-Goals

- Training/fitting regressors.
- Simulator, binding, precision-path, or predictor-filter code changes (adding `head_dim` to the predictor's exact-match filter is a noted follow-up, not done here).
- Any model other than `qwen3-a3b-30b-moe`.
- MoE (`moe_gating_linear`, `moe_gating_routing_topk`, `moe_shuffling`, `moe_grouped_gemm`) collection.
- Migrating legacy dense attention CSVs to carry `head_dim`.
- TORCH_SDPA re-collection.

## Constraints

- Branch `smatar/qwen3-30b-mi355-profiling` in worktree `/home/dn/Frontier-qwen3-profiling`, based on c68096c.
- No commits without explicit user approval (CLAUDE.md).
- Profiler writes CSVs only at the end of a run; each (block_size) cell is an independent invocation with its own output tree (template's checkpointing rule).
- `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` must be set explicitly (no `nvidia-smi` on ROCm).
- Qwen3 `linear_op` at `num_tokens=4000`, TP=1 faults the GPU: exclude that point.
- AITER has no block_size-32 prefill kernel; only {1,16} are profiled (also the request's set).
- `get_max_num_blocks` budgets against total GPU memory; the node must be exclusively allocated (no resident server).

## Success Criteria

- `01_plan.md` deliverables 1–5 are present and traceable (this file).
- After implement-loop: unit tests pass for the schema change and the new modules; dry-runs of all scripts print the expected two attention cells and the linear_op command.
- After the cluster runs: `python -m frontier.profiling.attention.dataset_tools validate …` reports PASS for `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/` with the expectations file; `02_run_manifest.md` is filled.
- Rejection conditions: any collected CSV lacking `head_dim`; `attention_backend != AITER` rows in canonical files; missing TP or block_size values; duplicate feature keys other than the ones the grid generator itself emits (or with a multiplicity different from the generator's); canonical `attention_combined.csv` row count ≠ standard + true-mixed rows.

## Explicit User Decisions

| Decision | User Answer | Resolved in Stage |
|----------|-------------|-------------------|
| Branch source for the AITER wrapper + sweep template (request said "off main", files live on feat/gptoss120b-mi355) | "Rebase onto feat branch" | 3 |
| Handling of 80 legacy dense CSVs once head_dim is required | "Strict, no migration" | 3 |
| MoE grouped GEMM has no AITER path in the profiler | "Defer MoE entirely" | 3 |
| Canonical (unsuffixed) attention files after the run | "AITER union, SDPA renamed" | 3 |
| Model | Qwen3-30B-A3B (`qwen3-a3b-30b-moe`), pre-selected in the request; 235B explicitly not chosen | 1 (request) |

## Unresolved Decisions

NONE

## Design

### D-1 Operator selection (REQ-1, REQ-2, REQ-10)

Manifest (resolved by `build_operator_manifest`, see `00_requirements_and_source_map.md`) and verdict:

| Operator | Profiler | (a) cost share | (b) extrapolation exposure today | (c) kernel path on this branch | Verdict |
|---|---|---|---|---|---|
| `attn_prefill` | attention | dominant TTFT term at ≥16k (O(n²) per layer × 48) | existing MI355X data: TORCH_SDPA, block 16 only, `max_model_len` 9472, no AITER rows → predictor extrapolates beyond 9k and never saw block 1 | AITER `mha_batch_prefill_func` available (`aiter_attention_wrapper.py`) | **INCLUDE** |
| `attn_decode` | attention | dominant TPOT term at long KV (KV read ∝ kv_cache_size × batch) | same as above | AITER `paged_attention_ragged` available | **INCLUDE** |
| `attn_kv_cache_save` | attention | small, memory-bound | same rows | real scatter into paged cache in the AITER wrapper | INCLUDE (free, same rows; predictor target `cache_write`) |
| `attn_pre_proj` (QKV GEMM + QK-norm) | linear_op | moderate: 16384×2048×5120 GEMM per layer at a 16k chunk | existing `linear_op.csv` stops at `num_tokens` 4096 → extrapolates for chunks > 4096 | vLLM/torch GEMM (hipBLASLt) in the same image; same library class production uses | **INCLUDE** (extend token grid to 16384) |
| `attn_post_proj` | linear_op | moderate (4096→2048 GEMM) | same | same | **INCLUDE** |
| `attn_rope` | linear_op | small | same | torch fallback RoPE (`FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`) | INCLUDE (free, same rows) |
| `input_layernorm`, `post_attention_layernorm`, `emb` | linear_op | small, memory-bound | same | torch | INCLUDE (free, same rows; `emb` is not a predictor target) |
| `add_attn_residual`, `add_ffn_residual` (profiling key `add`) | linear_op | — | — | not emitted for this model: `config.uses_fused_add_norm` → `_profile_add=False` (`linear_op_impl.py:765-767`, RMSNorm uses `fused_add_rms_norm`, so the add is inside the layernorm time); the existing MI355X `linear_op.csv` has no `time_stats.add.*` columns | EXCLUDE — fused into layernorm |
| `moe_grouped_gemm` | moe | dominant FFN term (8 × 2048×768×3 per token) | existing `moe.csv` stops at 4096 tokens | **no AITER path** (`moe_grouped_gemm_backend=vllm_fused` only) | **EXCLUDE — deferred (D3)** |
| `moe_gating_linear`, `moe_gating_routing_topk`, `moe_shuffling` | moe | small | same | vLLM `fused_topk` | EXCLUDE — deferred with MoE (D3) |
| `ffn/*`, `share_expert/*` | — | not bound for this model | — | — | EXCLUDE (not in manifest) |
| `comm/*`, `kv_transfer` | collectives / trace | network, not compute | — | — | EXCLUDE (out of compute-operator scope) |
| `lm_head_linear`, `mtp_fusion_proj` | linear_op (`--include_target_embedded_mtp`) | not in this model's manifest (MTP only) | — | — | EXCLUDE |

"Several" = the attention kernel pair (with cache save) and the attention-layer linear ops. Two profiler runs (attention, linear_op) produce them.

### D-2 Attention grid (REQ-3, REQ-4, REQ-5)

One sweep, `BACKENDS="AITER"`, two cells `qwen3-a3b-30b-moe__aiter__block{1,16}`, each covering `--num_tensor_parallel_workers 1 2 4 8`:

| Setting | Value | Source |
|---|---|---|
| `--attention_backend` | `AITER` | REQ-3 |
| `--block_size` | 1, then 16 (separate cells) | REQ-4; template `AITER_BLOCK_SIZES` |
| `--num_tensor_parallel_workers` | `1 2 4 8` (TP=8 replicates the 4 KV heads: valid per `model_config.py:440-454`) | REQ-5 |
| `--max_seq_len` / `--max_model_len` | 16384 / 16384 | REQ-5 ("≥16k"); template rationale for not raising further (5503 vs 2772 points) |
| `--batch_size_list` | `1 2 4 8 16 24 32 48 64 96 128 160 192 256 320 384 448 512` (`--min_batch_size 1 --max_batch_size 512`) | template |
| `--decode_kv_cache_size_list` | `128 512 1024 2048 4096 8192 16384` | template |
| `--enable_chunked_prefill_grid_search` | on (no `--fixed_chunked_prefill_size`) | REQ-5 |
| `--enable_true_mixed` | on; `--true_mixed_prefill_batch_sizes 1 2`; `--true_mixed_prefill_chunk_sizes 1024 4096 8192`; decode batch sizes = batch list; decode KV sizes = decode KV list; `--true_mixed_prefill_kv_cache_size 0` | REQ-5; template |
| `--precision BF16 --profile_method cuda_event --max_pipeline_parallel_size 1 --disable_ray --yes --device mi355x` | as listed | template; model dtype bf16 |
| `--num_gpus` | 4 (template's measured fault boundary for AITER; `AITER_NUM_GPUS` knob; 8 is retested in the smoke stage first) | template |
| Grid size per cell before memory filter | standard 2772 profiler attempts per TP (2646 prefill + 126 decode) = 11,088 rows per block, of which 2497 per TP are unique feature keys: 269 keys are measured twice and 3 keys three times (269 + 2×3 = 275 extra attempts per TP) because they are reachable by more than one grid path (`PREFILL_CHUNK_SIZE_SPACE` repeats 128/1024/4096/16384 at range seams, `utils/__init__.py:276-282`; a full prefill of length L coincides with chunk L at kv 0). Rows are kept as measured — the repo's own sanity check uses these pairs as reproducibility evidence (GPTOSS_TRUE_MIXED_BATCH_PROFILING.md check 1; observed spread in the gpt-oss AITER cell: median 0.6 %, p95 6.9 %, max 17.7 %). True-mixed 360 × 4 TP = 1,440 (no duplicates). | `get_attention_input_combinations`, `get_true_mixed_attention_input_combinations` computed on this checkout; gpt-oss `attention_aiter_block16.csv` (11,081 rows / 9,981 unique keys) |
| Memory filter | `is_under_memory_limit(max_num_blocks × block_size)` with `max_num_blocks = floor(mem_get_info()[1] × 0.9 / block_memory_total)`; KV/token/layer at TP1 = 2×4×128×2 B = 2 KB → 96 KB/token over 48 layers; TP2 48 KB; TP4/TP8 24 KB. With 0.9 × 288 GB the largest decode/true-mixed combos (e.g. batch 512 × KV 16384) drop at TP1 and TP2. `main.py` prints no per-TP standard post-filter count (only an aggregate true-mixed count), so provenance is recorded differently: the smoke stage logs `torch.cuda.mem_get_info()[1]` (`gpu_total_bytes`) into the run manifest, and `dataset_tools expected-grid` recomputes the surviving combos per (block_size, TP) offline with the same generator functions and `get_max_num_blocks` arithmetic. | `main.py:1833-1860`; `utils.get_max_num_blocks` (`utils/__init__.py:413-446`) |
| aiter JIT prewarm | every requested TP × block_size pair is warmed once, single-process, with the model-derived shard shape (`num_q_heads // TP`, `max(1, num_kv_heads // TP)`, `head_dim`) exactly as the template does for gpt-oss: Qwen3 → 8 warm calls ((32,4,128), (16,2,128), (8,1,128), (4,1,128) × blocks {1,16}). No dedupe on gqa_ratio: AITER's cache key is not documented in any source here, so the template's per-pair behaviour is kept | template `prewarm_aiter` (`profile_gptoss_attention_full_sweep.sh`), generalised (D-4) |

### D-3 Linear-op grid (REQ-9)

One run, in the same image (vllm present):

| Setting | Value | Source |
|---|---|---|
| `--models qwen3-a3b-30b-moe --is_moe` | skips dense MLP ops (MoE model) | `profile_mi355x.sh` |
| `--num_tensor_parallel_workers 1 2 4 8` | | REQ-5 |
| `--num_tokens_list` | `get_num_tokens_to_profile(16384)` (387 values) minus `4000` → 386 values, generated by `frontier.profiling.dataset_tools linear-token-grid --max-tokens 16384 --exclude 4000` | `utils.get_num_tokens_to_profile`; `profile_mi355x.sh` fault exclusion |
| `--precision BF16 --profile_method cuda_event --disable_ray --yes --device mi355x --num_gpus 8` | | template |
| env | `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1`, `CUDA_VISIBLE_DEVICES=HIP_VISIBLE_DEVICES=0..7` | cookbook gotchas #2, #4 (verified in `profile_mi355x.sh`) |
| Expected rows | 386 × 4 TP = 1,544 | grid |
| Output | `linear_op.csv` replaces the 4096-token file (superset grid, same schema and kernel path); the old file is renamed `linear_op_maxtokens4096.csv` for the same reason the SDPA trio is renamed | REQ-7 analogue |

### D-4 Scripts: reused vs new (REQ-8, REQ-11)

Reused unchanged: `frontier.profiling.attention.main`, `frontier.profiling.linear_op.main`, `aiter_attention_wrapper.py`, `backends/__init__.py`, `attention_dataset_contract.py`, `examples/profiling/profile_mi355x.sh` (pattern reference only).

Modified (behaviour-preserving):
- `profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh`: (i) `run_in_docker()` additionally bind-mounts `SHARED_ROOT` (new env var, default empty = no extra mount, so gpt-oss behaviour is unchanged) at the same path inside the container (`-v "$SHARED_ROOT":"$SHARED_ROOT"`) and forwards `WORK_DIR`, `COLLECT_DIR`, `LOG_DIR` in its `-e` list — today it mounts only `$REPO_ROOT` and `$AITER_CACHE_DIR` and forwards neither, which only works because the gpt-oss defaults keep `WORK_DIR`/`COLLECT_DIR`/`LOG_DIR` under `$REPO_ROOT`; (ii) `prewarm_aiter()` stops hard-coding gpt-oss dims (`NQ_TOTAL, NKV_TOTAL, HD = 64, 8, 64`) and calls `python -m frontier.profiling.attention.backends.aiter_prewarm --model "$model" --tps $AITER_TPS --block-sizes $AITER_BLOCK_SIZES` per model. Defaults (`MODELS`, `BACKENDS`, image) are unchanged, so gpt-oss runs are identical. The `DOCKER_IMAGE` default stays `v0.5.9`; the Qwen3 driver overrides it because only `v0.5.11-rocm700-mi35x` exists on the cluster nodes.
- `frontier/attention/families.py`, `frontier/profiling/attention/attention_wrapper.py`: head_dim (D-5).

New:
- `frontier/profiling/attention/backends/aiter_prewarm.py` — model-driven aiter template prewarm (importable, testable without GPU for spec derivation).
- `frontier/profiling/attention/dataset_tools.py` — `union` (canonical trio from per-cell files), `validate` (checklist as code, REQ-12), `expected-grid` (offline recompute of the memory-filtered grid), `linear-token-grid` (D-3), `manifest` (REQ-13). CLI via `python -m`.
- `profiling_knowledge/scripts/profile_qwen3_mi355x.sh` — node/container-side driver: `STAGE=smoke|attention|linear_op`; sets Qwen3 env for the sweep script (`MODELS=qwen3-a3b-30b-moe BACKENDS=AITER AITER_BLOCK_SIZES="1 16" AITER_TPS="1 2 4 8" MAX_SEQ_LEN=16384 DOCKER_IMAGE=lmsysorg/sglang:v0.5.11-rocm700-mi35x AITER_CACHE_DIR=/mnt/data/aiter-cache WORK_DIR=/opt/shared/frontier-qwen3-profiling/work COLLECT_DIR=/opt/shared/frontier-qwen3-profiling/collect LOG_DIR=/opt/shared/frontier-qwen3-profiling/logs`; the sweep's own default is `$WORK_DIR/logs`, so `LOG_DIR` must be passed explicitly and forwarded through `run_in_docker`'s `-e` list); the smoke stage writes `$LOG_DIR/smoke.log`, the linear stage `$LOG_DIR/linear_op.log`, runs linear_op with the D-3 command, supports `--dry-run`.
- `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` — `#SBATCH --partition=XAI --nodes=1 --ntasks=1 --gres=gpu:8 --time=24:00:00 --job-name=smatar-qwen3-prof --output=/opt/shared/frontier-qwen3-profiling/logs/%x-%j.out`; verifies the image with `docker image inspect`, runs `mkdir -p /mnt/data/aiter-cache` and logs `hostname`, `df -h /mnt/data`, then runs `profile_qwen3_mi355x.sh --docker` from the shared checkout.
- `profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh` — VM-side: `rsync` the worktree (excluding `.git` and `data/profiling/**/*.csv`) to `cluster:/opt/shared/frontier-qwen3-profiling/Frontier`, write `FRONTIER_COMMIT`, then `ssh cluster "sudo -u dn sbatch --export=STAGE=… …"`; `--dry-run` prints commands.
- `profiling_knowledge/scripts/slurm/ingest_qwen3_mi355x_results.sh` — VM-side: rsync `collect/compute/` into `data/profiling/compute/` and `/opt/shared/frontier-qwen3-profiling/logs/` (smoke.log, per-cell logs, linear_op.log, Slurm %x-%j.out files) into `logs/qwen3_mi355x/` in the worktree (git-ignored `logs/`), `git mv` the legacy trio and `linear_op.csv`, run `dataset_tools union`, `dataset_tools validate`, `dataset_tools manifest`.
- `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/expectations_qwen3_mi355x.json` — expectation values for the validator (D-6).
- `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/README.md` — runbook; `02_run_manifest.md` — filled by `dataset_tools manifest`.

### D-5 head_dim schema change (REQ-6)

- `DENSE_ATTENTION_FAMILY.required_profiling_feature_columns` gains `"head_dim"` immediately after `"n_kv_head"`.
- The three row builders in `attention_wrapper.py` are refactored to spread one shared `_attention_static_metadata()` dict (existing eight constants + `"head_dim": int(self._model_config.get_head_size())`); row-specific keys and column order are otherwise unchanged (`AGENTS.md` no-duplication gate). `get_head_size()` returns the explicit `head_dim` (128) or `embedding_dim // num_q_heads` (`model_config.py:455-463`); MLA is excluded because the dense family gates these builders (`AiterAttentionWrapper.supports_attention_family`).
- Enforcement points under "strict, no migration" (verified in repair iteration 1): `validate_attention_profiling_dataframe(…, DENSE_ATTENTION_FAMILY)` runs at profiler write time (`main.py:886`) and inside `dataset_tools validate`; the predictor and shared-manager load paths call it only for `LATENT_MLA_ATTENTION_FAMILY` (`sklearn_execution_time_predictor.py:1233-1238`, `shared_prediction_model_manager.py:1682-1686`) and the dense loader filters columns without schema validation (`sklearn_execution_time_predictor.py:1222-1228`). Consequently the 80 legacy dense CSVs (none with `head_dim`) keep loading in the simulator, training CLI, `smoke_simulator_dense_csv.sh` and e2e baselines; they fail only when passed through `dataset_tools validate` or any future caller of the dense validator. This is documented in the runbook and `docs/profiling/README.md`; adding a dense load-time gate is a follow-up decision, not part of this plan.
- Synthetic-fixture unit tests listed under "Schema-pinning tests" get `head_dim` added to their dense rows and pinned tuples.
- The predictor's exact-match filter is not changed (non-goal); noted as follow-up.

### D-6 Output files and schema (REQ-7, REQ-13)

Cluster (shared): `/opt/shared/frontier-qwen3-profiling/{Frontier,work,collect,logs}`. Each attention cell writes to `work/qwen3-a3b-30b-moe__aiter__block{N}/compute/mi355x/qwen3-a3b-30b-moe/{attention,attention_true_mixed,attention_combined}.csv` and its own `compute/mi355x/attention_config.yaml` (so the device-level yaml overwrite in `main.py:1921` cannot clobber another cell). `collect` copies them to `collect/compute/mi355x/qwen3-a3b-30b-moe/attention{,_true_mixed,_combined}_aiter_block{N}.csv` (template naming). linear_op writes `collect/compute/mi355x/qwen3-a3b-30b-moe/linear_op.csv`.

Repo after ingest, `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/`:

| File | Content |
|---|---|
| `attention_aiter_block1.csv`, `attention_true_mixed_aiter_block1.csv`, `attention_combined_aiter_block1.csv` | cell block 1 |
| `attention_aiter_block16.csv`, `attention_true_mixed_aiter_block16.csv`, `attention_combined_aiter_block16.csv` | cell block 16 |
| `attention.csv` | union of the two `attention_aiter_block*.csv` (standard rows, both block sizes; exact-match filter selects) |
| `attention_true_mixed.csv` | union of the two true-mixed files |
| `attention_combined.csv` | union of the two combined files (= standard + true-mixed) — the file the simulator is pointed at |
| `attention_sdpa_block16.csv`, `attention_true_mixed_sdpa_block16.csv`, `attention_combined_sdpa_block16.csv` | legacy TORCH_SDPA trio, `git mv` |
| `linear_op.csv` | new 16k-token run; `linear_op_maxtokens4096.csv` = legacy |
| `moe.csv` | untouched (MoE deferred) |
| `PROVENANCE.md` | copy of `02_run_manifest.md` summary |

Contract safety: the canonical `attention.csv` carries `is_mixed_batch`/`mode`/`total_tokens` (standard writer, `attention_wrapper.py:388-402`), so `enforce_mixed_attention_input_contract` passes with `attention_true_mixed.csv`/`attention_combined.csv` beside it; the contract inspects the unsuffixed mixed siblings for any attention input path in the directory, and every per-cell suffixed file also carries the marker columns, so pointing the simulator at a suffixed file passes too. Union keeps the column order of the block-16 file and asserts identical column sets before concatenation.

Attention schema = existing 61-column standard schema (72 in combined) + `head_dim` (int, 128 for every row). linear_op schema unchanged.

### D-7 Validation checklist (REQ-12) — implemented by `dataset_tools validate`, also run by hand

1. Schema: `validate_attention_profiling_dataframe(df, DENSE_ATTENTION_FAMILY, measurement_type="CUDA_EVENT")` passes for all nine attention files; `head_dim` column present and == 128 everywhere; `n_embd 2048, n_q_head 32, n_kv_head 4`; `profiling_precision BF16`; `measurement_type CUDA_EVENT`; `attention_backend == "AITER"` in all AITER/canonical files; no legacy columns missing vs the block-16 file (no schema drift between cells).
2. Value sets: `block_size` == {1} / {16} per cell and {1,16} in canonical; `num_tensor_parallel_workers` == {1,2,4,8}; `max_model_len` == {16384}. Standard rows (`mode == even`): decode `batch_size` ⊆ batch list, decode `kv_cache_size` ⊆ decode KV list, prefill rows have `batch_size == 1`. True-mixed rows (`mode == true_mixed`, `is_true_mixed_batch`): `num_prefill_seqs` ∈ {1,2}, `prefill_seq_lens` values ⊆ {1024,4096,8192}, `decode_batch_size` ⊆ batch list, `decode_kv_cache_sizes` values ⊆ decode KV list, `batch_size == total_batch_size == num_prefill_seqs + decode_batch_size ≤ 128` (`true_mixed_batch_input.py:104-123`; the list check is NOT applied to true-mixed `batch_size`).
3. Grid coverage per (block, TP): the validator computes `expected_attention_grid(expectation, gpu_total_bytes, block_size, tp)` — the standard and true-mixed combos that survive `is_under_memory_limit` for `max_num_blocks = floor(gpu_total_bytes × 0.9 / block_memory_total)` — and requires the CSV's feature-key set to equal it exactly: every surviving prefill chunk size of `get_attention_prefill_chunk_sizes_to_profile(16384)`, every full-prefill length of `get_seq_lengths_to_profile(16384)`, every surviving (batch, kv) decode combo, every surviving true-mixed combo. Combos that were dropped by the memory filter are listed in the report as expected gaps; any other missing or extra row is a failure. `gpu_total_bytes` comes from the run manifest (recorded by the smoke stage).
4. Row counts: per-cell `attention_aiter_blockN` rows == surviving profiler attempts (duplicates included, no dedupe anywhere in the pipeline); `attention_combined_aiter_blockN` rows == `attention_aiter_blockN` + `attention_true_mixed_aiter_blockN`; canonical files == sum of the two cells; counts recorded in the manifest.
5. Quality: no NaN in `time_stats.*.median`; `attn_prefill.median > 0` for prefill rows, `attn_decode.median > 0` for decode rows; `time_stats.*.count == 5`; soft check — decode median non-decreasing in `kv_cache_size` at fixed (TP, batch) and in `batch_size` at fixed (TP, kv) (warn, not fail); block-1 vs block-16 decode at the same shape within a plausible band (warn).
6. Duplicates: duplicate feature keys (`block_size, num_tensor_parallel_workers, batch_size, prefill_chunk_size, kv_cache_size, is_prefill, mode`) are allowed only where `expected_attention_grid` predicts them, with exactly the predicted multiplicity; within each duplicate group the `attn_prefill`/`attn_decode` medians must agree within `duplicate_warn_rel_spread` (default 0.10, warn) and `duplicate_fail_rel_spread` (default 0.30, fail) — thresholds live in the expectations JSON and are set from the gpt-oss AITER observation (max 17.7 %); any duplicate key outside the predicted set is a failure (it would indicate an accidental double collection); canonical files contain exactly the per-cell rows (no dedupe, no extra rows).
7. Filesystem: no symlinks under the model dir (`find -type l`); the `deepseek-r1-0528 -> deepseek-v3` symlink elsewhere is untouched; every file's sha256 recorded; legacy trio renamed (no unsuffixed TORCH_SDPA rows).
8. Contract: `enforce_mixed_attention_input_contract(<dir>/attention.csv, columns)` raises nothing.
9. linear_op: rows == 386 × 4 = 1,544; `num_tokens` set == expected 386 values at every TP; TP set == {1,2,4,8}; `n_head 32, n_kv_head 4, n_embd 2048, use_qk_norm True`; `time_stats.*` scopes present exactly for {emb, input_layernorm, attn_pre_proj, attn_rope, attn_post_proj, post_attention_layernorm} and no `time_stats.add.*` (fused add+RMSNorm); NaN pattern scoped per operator: `attn_pre_proj`/`attn_rope`/`attn_post_proj` medians non-null on every row; replicated ops (`emb`, `input_layernorm`, `post_attention_layernorm`) non-null on TP=1 rows and NaN on TP>1 rows because `linear_op/main.py:581-586` splits replicated ops onto TP=1 (the existing MI355X file shows exactly this: 0 NaN at TP1, 258 NaN per TP>1); `attn_pre_proj.median` increasing with `num_tokens` (soft).
10. Log checks: each cell log contains no "OutOfMemoryError|Memory access fault|BrokenProcessPool|Traceback"; the aiter prewarm log lists all 8 (TP, block) warm calls; the smoke log contains the `gpu_total_bytes=` line copied into the manifest.

## Files / Modules

| Action | Path | Purpose |
|--------|------|---------|
| modify | `frontier/attention/families.py` | add `"head_dim"` to dense required columns (D-5) |
| modify | `frontier/profiling/attention/attention_wrapper.py` | record `head_dim` in the three row builders (D-5) |
| create | `frontier/profiling/attention/backends/aiter_prewarm.py` | model-driven aiter template prewarm (D-2, D-4) |
| create | `frontier/profiling/attention/dataset_tools.py` | union / validate / linear-token-grid / manifest (D-3, D-6, D-7) |
| modify | `profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh` | `prewarm_aiter()` delegates to `aiter_prewarm` (D-4) |
| create | `profiling_knowledge/scripts/profile_qwen3_mi355x.sh` | Qwen3 driver (smoke / attention / linear_op) (D-4) |
| create | `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` | Slurm job (D-4) |
| create | `profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh` | VM-side rsync + sbatch (D-4) |
| create | `profiling_knowledge/scripts/slurm/ingest_qwen3_mi355x_results.sh` | VM-side rsync back, rename, union, validate, manifest (D-4, D-6) |
| create | `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/expectations_qwen3_mi355x.json` | validator expectations (D-7) |
| create | `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/README.md` | runbook + strict-head_dim consequences |
| create | `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/02_run_manifest.md` | filled by `dataset_tools manifest` (REQ-13) |
| modify | `profiling_knowledge/README.md`, `docs/profiling/README.md` | index entry; head_dim requirement note |
| modify | `tests/unit/test_attention_family_specs.py`, `tests/unit/test_attention_family_spec_data.py` | pinned tuples gain `head_dim` |
| modify | `tests/unit/test_attention_main_true_mixed_contract.py`, `test_attention_tp_effective_mapping.py`, `test_execution_time_predictor_max_tokens_budget.py`, `test_shared_prediction_model_manager_eager_attention_decode.py`, `test_spec_decode_predictor_verify_prefill_path.py`, `test_mha_stage2_profile_modeling.py`, `test_mqa_stage2_profile_modeling.py` | dense fixtures gain `head_dim` |
| create | `tests/unit/test_attention_head_dim_schema.py` | new tests for D-5 |
| create | `tests/unit/test_aiter_prewarm_specs.py` | new tests for `aiter_prewarm` |
| create | `tests/unit/test_attention_dataset_tools.py` | new tests for `dataset_tools` |
| create | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | shell syntax + dry-run contracts for the new scripts |
| rename (ingest) | `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/attention{,_true_mixed,_combined}.csv` → `*_sdpa_block16.csv`; `linear_op.csv` → `linear_op_maxtokens4096.csv` | legacy files (D-6) |
| create (ingest) | `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/attention*_aiter_block{1,16}.csv`, canonical trio, `linear_op.csv`, `PROVENANCE.md` | collected data (D-6) |

## Classes / Functions / Signatures

```python
# file: frontier/attention/families.py  (DENSE_ATTENTION_FAMILY, edit in place)
required_profiling_feature_columns=(
    "measurement_type",
    "attention_backend",
    "n_q_head",
    "n_kv_head",
    "head_dim",
    "block_size",
    "num_tensor_parallel_workers",
    "max_model_len",
    "batch_size",
    "prefill_chunk_size",
    "kv_cache_size",
    "is_prefill",
),
```

```python
# file: frontier/profiling/attention/attention_wrapper.py
class AttentionWrapper:
    def _attention_static_metadata(self) -> dict:
        """Model/shard constants shared by every profiling row (standard, mixed, true-mixed).

        Replaces the three hand-copied dict fragments in profile(), profile_mixed(), profile_true_mixed().
        """
        return {
            "n_embd": self._model_config.embedding_dim,
            "n_q_head": self._model_config.num_q_heads,
            "n_kv_head": self._model_config.num_kv_heads,
            "head_dim": int(self._model_config.get_head_size()),
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "attention_backend": self._attention_backend,
        }
    # profile(), profile_mixed(), profile_true_mixed(): result = {"time_stats": time_stats, **self._attention_static_metadata(), <row-specific keys unchanged>}
```

```python
# file: frontier/profiling/attention/backends/aiter_prewarm.py
from dataclasses import dataclass
from typing import Sequence

_AITER_PARTITION_SIZE_ROCM = 256  # same constant as aiter_attention_wrapper

@dataclass(frozen=True)
class AiterTemplateSpec:
    """One aiter decode-kernel warm call for a (TP, block_size) pair with the model's shard shape."""
    tensor_parallel_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int

    @property
    def gqa_ratio(self) -> int:
        """num_q_heads // num_kv_heads for this TP shard."""
        # body implemented in implement-loop (see Task Breakdown)

def aiter_template_specs(
    model_name: str,
    tensor_parallel_sizes: Sequence[int],
    block_sizes: Sequence[int],
) -> tuple[AiterTemplateSpec, ...]:
    """Derive one spec per requested (TP, block_size) with shard shapes from ModelConfig.from_model_name
    (num_q_heads // tp, get_num_kv_heads(ParallelConfig(tp)), get_head_size()); no dedupe — mirrors the template."""
    # body implemented in implement-loop (see Task Breakdown)

def prewarm_aiter_templates(specs: Sequence[AiterTemplateSpec], device: str = "cuda:0") -> None:
    """Run paged_attention_ragged once per spec so the JIT template is compiled and cached (GPU required)."""
    # body implemented in implement-loop (see Task Breakdown)

def main(argv: Sequence[str] | None = None) -> int:
    """CLI: --model NAME --tps 1 2 4 8 --block-sizes 1 16 [--dry-run] [--device cuda:0]; --dry-run prints specs and exits 0."""
    # body implemented in implement-loop (see Task Breakdown)
```

```python
# file: frontier/profiling/attention/dataset_tools.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import pandas as pd

@dataclass(frozen=True)
class AttentionDatasetExpectation:
    """Expected constants and grids for one collected attention dataset directory (loaded from JSON)."""
    model_name: str
    device: str
    attention_backend: str
    block_sizes: tuple[int, ...]
    tensor_parallel_sizes: tuple[int, ...]
    max_model_len: int
    n_embd: int
    n_q_head: int
    n_kv_head: int
    head_dim: int
    batch_sizes: tuple[int, ...]
    decode_kv_cache_sizes: tuple[int, ...]
    true_mixed_prefill_batch_sizes: tuple[int, ...]
    true_mixed_prefill_chunk_sizes: tuple[int, ...]
    profiling_precision: str = "BF16"
    measurement_type: str = "CUDA_EVENT"
    per_cell_suffix: str = "aiter_block{block_size}"
    linear_op_max_tokens: int = 16384
    linear_op_excluded_tokens: tuple[int, ...] = (4000,)
    duplicate_warn_rel_spread: float = 0.10
    duplicate_fail_rel_spread: float = 0.30
    gpu_memory_utilization: float = 0.9        # must equal get_max_num_blocks' default used by attention/main.py
    max_pipeline_parallel_size: int = 1        # must equal the profiler's --max_pipeline_parallel_size
    # operators that must come from build_operator_manifest(model) profiling names
    selected_manifest_operators: tuple[str, ...] = (
        "attn_kv_cache_save", "attn_prefill", "attn_decode",
        "input_layernorm", "post_attention_layernorm", "emb",
    )
    # attention-layer linear ops: predictor inputs from attention_tp_policy.ATTENTION_LINEAR_OPS, not manifest members
    selected_attention_linear_operators: tuple[str, ...] = ("attn_pre_proj", "attn_rope", "attn_post_proj")
    # replicated memory ops are emitted on TP=1 rows only (linear_op/main.py:581-586 split_replicated_result)
    replicated_tp1_only_operators: tuple[str, ...] = ("emb", "input_layernorm", "post_attention_layernorm")

    @classmethod
    def from_json(cls, path: Path) -> "AttentionDatasetExpectation":
        """Load and type-check the expectation JSON."""
        # body implemented in implement-loop (see Task Breakdown)

@dataclass
class ValidationReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)

    def render_markdown(self) -> str:
        """Render failures, warnings, row counts and checksums as Markdown."""
        # body implemented in implement-loop (see Task Breakdown)

def union_block_files(per_cell_files: Sequence[Path], output_file: Path) -> pd.DataFrame:
    """Concatenate per-block-size CSVs with identical column sets into one canonical file; raise on column mismatch."""
    # body implemented in implement-loop (see Task Breakdown)

def linear_token_grid(max_tokens: int, excluded: Sequence[int]) -> list[int]:
    """get_num_tokens_to_profile(max_tokens) minus excluded, in the generator's own (descending) order."""
    # body implemented in implement-loop (see Task Breakdown)

@dataclass(frozen=True)
class ExpectedAttentionGrid:
    """Feature keys expected in one (block_size, TP) cell after the profiler's memory filter."""
    block_size: int
    tensor_parallel_size: int
    max_num_blocks: int
    standard_key_multiplicity: dict[tuple, int]   # (prefill_chunk_size, kv_cache_size, batch_size, is_prefill) -> attempts (1, 2 or 3)
    true_mixed_keys: frozenset[tuple]             # (num_prefill_seqs, prefill_chunk, decode_batch_size, decode_kv_cache_size)
    dropped_standard_keys: frozenset[tuple]
    dropped_true_mixed_keys: frozenset[tuple]

    @property
    def standard_attempts(self) -> int:
        """Surviving profiler attempts = expected per-cell standard row count (duplicates included)."""
        # body implemented in implement-loop (see Task Breakdown)

def expected_attention_grid(
    expectation: AttentionDatasetExpectation,
    gpu_total_bytes: int,
    block_size: int,
    tensor_parallel_size: int,
) -> ExpectedAttentionGrid:
    """Recompute the profiler's surviving grid offline: same generators as attention/main.py, same
    max_num_blocks arithmetic as utils.get_max_num_blocks (gpu_memory_utilization and max_pipeline_parallel_size taken
    from the expectation, which pins the profiler's command-line values) but fed gpu_total_bytes instead of mem_get_info()."""
    # body implemented in implement-loop (see Task Breakdown)

def validate_attention_dataset_dir(
    dataset_dir: Path,
    expectation: AttentionDatasetExpectation,
    gpu_total_bytes: int,
    cell_logs: dict[int, Path] | None = None,
) -> ValidationReport:
    """Run checklist items 1-8 and 10 (D-7) over per-cell and canonical attention files; item 3 uses expected_attention_grid."""
    # body implemented in implement-loop (see Task Breakdown)

def validate_linear_op_file(linear_op_csv: Path, expectation: AttentionDatasetExpectation) -> ValidationReport:
    """Run checklist item 9 (D-7)."""
    # body implemented in implement-loop (see Task Breakdown)

def write_run_manifest(
    dataset_dir: Path,
    reports: Sequence[ValidationReport],
    run_metadata: dict[str, str],
    output_file: Path,
) -> None:
    """Write 02_run_manifest.md: image, commit, node, wall time, row counts, sha256 per file."""
    # body implemented in implement-loop (see Task Breakdown)

def main(argv: Sequence[str] | None = None) -> int:
    """CLI subcommands: union | validate | expected-grid | linear-token-grid | manifest (see README for exact flags)."""
    # body implemented in implement-loop (see Task Breakdown)
```

```bash
# file: profiling_knowledge/scripts/profile_qwen3_mi355x.sh
# usage: profile_qwen3_mi355x.sh --stage {smoke|attention|linear_op} [--docker] [--dry-run]
# env knobs (defaults): MODEL=qwen3-a3b-30b-moe DEVICE=mi355x IMAGE=lmsysorg/sglang:v0.5.11-rocm700-mi35x
#   AITER_TPS="1 2 4 8" AITER_BLOCK_SIZES="1 16" AITER_NUM_GPUS=4 MAX_SEQ_LEN=16384
#   WORK_DIR=/opt/shared/frontier-qwen3-profiling/work COLLECT_DIR=/opt/shared/frontier-qwen3-profiling/collect
#   SHARED_ROOT=/opt/shared/frontier-qwen3-profiling      (bind-mounted 1:1 into the container by run_in_docker; WORK_DIR/COLLECT_DIR/LOG_DIR must live under it)
#   LOG_DIR=/opt/shared/frontier-qwen3-profiling/logs   (overrides the sweep's default $WORK_DIR/logs; all stages log here)
#   smoke stage: WORK_DIR=$SHARED_ROOT/work/smoke (own cell namespace so its 512-token block16 file can never make the full block16 cell be skipped)
#   AITER_CACHE_DIR=/mnt/data/aiter-cache LINEAR_MAX_TOKENS=16384 LINEAR_EXCLUDED_TOKENS=4000
# log files: $LOG_DIR/smoke.log (smoke stage; contains the gpu_total_bytes= line), $LOG_DIR/<cell>.log per attention cell,
#            $LOG_DIR/linear_op.log; the Slurm stdout/stderr goes to $LOG_DIR/%x-%j.out and is kept alongside.
# smoke: MAX_SEQ_LEN=512, TP 8 only, block 16, batch list "1 8", num_gpus 8 → asserts head_dim in output header and attention_backend==AITER;
#        prints `gpu_total_bytes=<torch.cuda.mem_get_info()[1]>` and `rocm-smi --showmemuse` and aborts if any GPU has >1 GB in use.

# file: profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch
# env: STAGE (required), FRONTIER_ROOT=/opt/shared/frontier-qwen3-profiling/Frontier, IMAGE
# file: profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh
# usage: submit_qwen3_mi355x_profiling.sh --stage {smoke|attention|linear_op} [--dry-run]
# file: profiling_knowledge/scripts/slurm/ingest_qwen3_mi355x_results.sh
# usage: ingest_qwen3_mi355x_results.sh [--dry-run]   (rsync back → git mv legacy → union → validate → manifest)
```

## Task Breakdown

- TASK-1: head_dim recorded + required
  - Depends on: none
  - Files: `frontier/attention/families.py`, `frontier/profiling/attention/attention_wrapper.py` (`_attention_static_metadata` + three call sites), `tests/unit/test_attention_head_dim_schema.py`, the nine schema-pinning test files
  - REQs covered: REQ-6
  - Source: Dense attention family spec; Attention wrapper row builders; Profiling ModelConfig

- TASK-2: model-driven aiter prewarm + sweep-script delegation
  - Depends on: none
  - Files: `frontier/profiling/attention/backends/aiter_prewarm.py`, `profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh`, `tests/unit/test_aiter_prewarm_specs.py`
  - REQs covered: REQ-8, REQ-3 (prewarm shard shapes derived from the model config)
  - Source: gpt-oss sweep template `prewarm_aiter`; AITER dense wrapper `_AITER_PARTITION_SIZE_ROCM`; Profiling ModelConfig `get_num_kv_heads`/`get_head_size`

- TASK-3: dataset_tools (union, validate, linear-token-grid, manifest) + expectations JSON
  - Depends on: TASK-1 (validator requires head_dim)
  - Files: `frontier/profiling/attention/dataset_tools.py`, `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/expectations_qwen3_mi355x.json`, `tests/unit/test_attention_dataset_tools.py`
  - REQs covered: REQ-1, REQ-2, REQ-7, REQ-9 (token grid), REQ-10, REQ-12, REQ-13
  - Source: Operator binding; Grid generators; Attention dataset contract; Predictor attention loader; MI355X collection driver (4000 exclusion)

- TASK-4: Qwen3 driver script (smoke / attention / linear_op)
  - Depends on: TASK-2, TASK-3
  - Files: `profiling_knowledge/scripts/profile_qwen3_mi355x.sh`, `tests/unit/test_qwen3_mi355x_profiling_scripts.py`
  - REQs covered: REQ-3, REQ-4, REQ-5, REQ-9, REQ-10 (no MoE stage), REQ-11 (image, shared-root mount and env forwarding)
  - Source: gpt-oss sweep template; linear_op profiler; MI355X collection driver; Cluster probe

- TASK-5: Slurm submit / sbatch / ingest scripts
  - Depends on: TASK-4
  - Files: `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch`, `submit_qwen3_mi355x_profiling.sh`, `ingest_qwen3_mi355x_results.sh`, `tests/unit/test_qwen3_mi355x_profiling_scripts.py`
  - REQs covered: REQ-11, REQ-7 (rename/union in ingest), REQ-13
  - Source: Cluster probe; project memory (sbatch + `sudo -u dn`); amd-playground sbatch template pattern

- TASK-6: Documentation
  - Depends on: TASK-1..5
  - Files: `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/README.md`, `profiling_knowledge/README.md`, `docs/profiling/README.md`
  - REQs covered: REQ-6 (enforcement points), REQ-10 (MoE deferral recorded), REQ-14
  - Source: Request; user decisions

- TASK-7: Smoke run on the cluster
  - Depends on: TASK-5
  - Files: none (produces `$LOG_DIR/%x-%j.out`, `$LOG_DIR/smoke.log`, `$SHARED_ROOT/work/smoke/...`)
  - REQs covered: REQ-3, REQ-6, REQ-11
  - Source: Cluster probe; Attention profiler entry point

- TASK-8: Full attention sweep (2 cells) and linear_op run
  - Depends on: TASK-7 passed
  - Files: cluster `collect/compute/mi355x/qwen3-a3b-30b-moe/*`
  - REQs covered: REQ-3, REQ-4, REQ-5, REQ-9
  - Source: D-2, D-3

- TASK-9: Ingest, rename, union, validate, manifest
  - Depends on: TASK-8
  - Files: `data/profiling/compute/mi355x/qwen3-a3b-30b-moe/*` (renames + new files), `02_run_manifest.md`, `PROVENANCE.md`
  - REQs covered: REQ-7, REQ-12, REQ-13
  - Source: D-6, D-7

## Tests Per Task

| Task | Test file | Test name | What it verifies |
|------|-----------|-----------|-----------------|
| TASK-1 | `tests/unit/test_attention_head_dim_schema.py` | `test_dense_required_columns_include_head_dim_after_n_kv_head` | tuple order and membership |
| TASK-1 | `tests/unit/test_attention_head_dim_schema.py` | `test_validate_dense_frame_rejects_missing_head_dim` | strict rejection message names `head_dim` |
| TASK-1 | `tests/unit/test_attention_head_dim_schema.py` | `test_attention_static_metadata_records_head_dim_128_for_qwen3` | `object.__new__(AttentionWrapper)` with `_model_config = ModelConfig.from_model_name("qwen3-a3b-30b-moe")`, `_block_size`, `_parallel_config`, `_max_model_len`, `_attention_backend` set → `_attention_static_metadata()` has the nine keys and `head_dim == 128` (CPU only) |
| TASK-1 | `tests/unit/test_attention_head_dim_schema.py` | `test_attention_static_metadata_falls_back_to_embd_over_heads` | a ModelConfig without explicit head_dim yields `embedding_dim // num_q_heads` |
| TASK-1 | `tests/unit/test_attention_family_specs.py` | `test_profiling_schema_columns_are_derived_from_family_spec` (updated) | pinned tuple includes `head_dim` |
| TASK-1 | the nine fixture tests | existing names (updated) | fixtures carry `head_dim`, suites stay green |
| TASK-2 | `tests/unit/test_aiter_prewarm_specs.py` | `test_qwen3_specs_cover_every_tp_block_pair_with_shard_shapes` | TP {1,2,4,8} × blocks {1,16} → 8 specs; (nq, nkv) = (32,4), (16,2), (8,1), (4,1); head_dim 128 |
| TASK-2 | `tests/unit/test_aiter_prewarm_specs.py` | `test_gptoss_specs_match_previous_hardcoded_shapes` | gpt-oss-120b yields nq 64/TP, nkv 8/TP, hd 64 → behaviour preserved |
| TASK-2 | `tests/unit/test_aiter_prewarm_specs.py` | `test_cli_dry_run_prints_specs_without_torch_cuda` | `main(["--model","qwen3-a3b-30b-moe","--tps","1","8","--block-sizes","1","16","--dry-run"]) == 0` |
| TASK-2 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_gptoss_sweep_prewarm_delegates_to_module_and_defaults_unchanged` | script contains `aiter_prewarm` call; `MODELS` default still `openai/gpt-oss-120b`; `bash -n` passes |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_qwen3_manifest_matches_expectation_selection` | `build_operator_manifest(qwen3-a3b-30b-moe)` resolves dense_attention/gqa + memory/replicated + moe/routed, no share_expert/ffn; `selected_manifest_operators` ⊆ manifest profiling names; `selected_attention_linear_operators` == `attention_tp_policy.ATTENTION_LINEAR_OPS`; neither set contains a `moe_*` or `add` name (REQ-1, REQ-2, REQ-10) |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_union_block_files_concatenates_and_preserves_columns` | union of two synthetic per-cell CSVs; row count = sum; column order kept |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_union_block_files_rejects_column_mismatch` | raises on differing column sets |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_linear_token_grid_excludes_4000_and_matches_generator` | list equals `[t for t in get_num_tokens_to_profile(16384) if t != 4000]` (386 values, descending, `4000` absent) |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_validate_passes_on_conforming_synthetic_dataset` | synthetic dir with head_dim, both blocks, all TPs → `passed` |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_validate_fails_on_missing_head_dim_wrong_backend_missing_tp_unexpected_duplicate_symlink` (parametrised) | each defect produces a named failure; an *expected* duplicate pair with 5 % spread passes, 20 % spread warns, 40 % spread fails |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_validate_combined_equals_standard_plus_true_mixed` | count mismatch fails |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_expected_grid_matches_generators_and_drops_over_budget_combos` | with a very large `gpu_total_bytes` the grid has 2772 standard attempts over 2497 unique keys (multiplicity distribution exactly {1: 2225, 2: 269, 3: 3}) + 360 true-mixed keys per TP; with `gpu_total_bytes = 288 GiB` the TP1 grid drops exactly the combos where `is_under_memory_limit` is false (e.g. batch 512 × KV 16384) and TP4 drops none |
| TASK-3 | `tests/unit/test_attention_dataset_tools.py` | `test_write_run_manifest_includes_checksums_and_counts` | manifest markdown contains sha256 and row counts |
| TASK-4 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_qwen3_driver_dry_run_attention_prints_two_aiter_cells_and_no_sdpa` | dry-run output has exactly two cells (`block1`, `block16`), each with `--attention_backend AITER`, `--num_tensor_parallel_workers 1 2 4 8`, `--block_size {1,16}`, `--max_seq_len 16384 --max_model_len 16384`, `--enable_chunked_prefill_grid_search`, `--enable_true_mixed`, `--precision BF16`, `--profile_method cuda_event`; no `frontier.profiling.attention.main` command line carries `--attention_backend TORCH_SDPA` (the reused script's banner text may still mention SDPA) |
| TASK-4 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_qwen3_driver_dry_run_linear_op_uses_386_token_grid_and_is_moe` | command contains `--is_moe`, 386 tokens, no `4000`, rope fallback exported |
| TASK-4 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_qwen3_driver_has_no_moe_stage` | `--stage moe` exits 2 |
| TASK-5 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_sbatch_declares_xai_partition_8_gpus_and_shared_paths` | header directives and `/opt/shared/frontier-qwen3-profiling` |
| TASK-5 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_submit_dry_run_prints_rsync_excludes_and_sudo_u_dn_sbatch` | rsync excludes `.git` and CSVs; `sudo -u dn sbatch` present |
| TASK-5 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_ingest_dry_run_renames_legacy_trio_and_runs_union_validate_manifest` | rsync sources are `collect/compute/` and `logs/` under `/opt/shared/frontier-qwen3-profiling`, destinations `data/profiling/compute/` and `logs/qwen3_mi355x/`; `git mv` targets `*_sdpa_block16.csv`, `linear_op_maxtokens4096.csv`; `validate` receives `--gpu-total-bytes` parsed from `logs/qwen3_mi355x/smoke.log`; tool subcommands invoked in order union → validate → manifest |
| TASK-4 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_qwen3_driver_forwards_shared_paths_into_docker` | dry-run of `--stage attention --docker` shows `-v /opt/shared/frontier-qwen3-profiling:/opt/shared/frontier-qwen3-profiling` and `-e WORK_DIR=… -e COLLECT_DIR=… -e LOG_DIR=…` on the docker command; smoke → `WORK_DIR=…/work/smoke` and `smoke.log`, linear → `linear_op.log` |
| TASK-2 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_gptoss_sweep_docker_helper_unchanged_without_shared_root` | with `SHARED_ROOT` unset the dry-run docker command has exactly the original two mounts (`$REPO_ROOT`, aiter cache) |
| TASK-6 | `tests/unit/test_qwen3_mi355x_profiling_scripts.py` | `test_docs_mention_head_dim_requirement_and_enforcement_points` | `docs/profiling/README.md` and task README name `head_dim` as required and state it is enforced at profiler write time and by `dataset_tools validate` only |
| TASK-5 | verification command | `git merge-base --is-ancestor c68096c HEAD` | branch is based on feat/gptoss120b-mi355 (REQ-14) |
| TASK-7 | cluster log | smoke assertions in `profile_qwen3_mi355x.sh --stage smoke` | output CSV header has `head_dim`; `attention_backend == AITER`; aiter import + signatures OK; 8-GPU run completes; `gpu_total_bytes=` line present |
| TASK-8 | cluster logs | `profile_cell` "ok" lines for both cells; linear_op log ends cleanly | no OOM / memory fault / BrokenProcessPool |
| TASK-9 | `dataset_tools validate` | PASS report | D-7 items 1–10 |

## Verification Commands

```bash
# in /home/dn/Frontier-qwen3-profiling
git merge-base --is-ancestor c68096c HEAD && echo 'branch based on feat/gptoss120b-mi355 (REQ-14)'
python -m pytest tests/unit/test_attention_head_dim_schema.py tests/unit/test_attention_family_specs.py \
  tests/unit/test_attention_family_spec_data.py tests/unit/test_attention_main_true_mixed_contract.py \
  tests/unit/test_attention_tp_effective_mapping.py tests/unit/test_execution_time_predictor_max_tokens_budget.py \
  tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py \
  tests/unit/test_spec_decode_predictor_verify_prefill_path.py tests/unit/test_mha_stage2_profile_modeling.py \
  tests/unit/test_mqa_stage2_profile_modeling.py tests/unit/test_aiter_prewarm_specs.py \
  tests/unit/test_attention_dataset_tools.py tests/unit/test_qwen3_mi355x_profiling_scripts.py \
  tests/unit/test_examples_profiling_contracts.py tests/unit/test_profiling_dataset_metadata_contract.py -q
python -m pytest tests/unit -q -x --deselect tests/unit/test_mla_predictor_training_integration.py   # full unit sweep
for f in profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh profiling_knowledge/scripts/profile_qwen3_mi355x.sh \
         profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh \
         profiling_knowledge/scripts/slurm/ingest_qwen3_mi355x_results.sh; do bash -n "$f" || { echo "SYNTAX FAIL: $f"; failed=1; }; done; exit "${failed:-0}"
PYTHONPATH=. python -m frontier.profiling.attention.backends.aiter_prewarm --model qwen3-a3b-30b-moe --tps 1 2 4 8 --block-sizes 1 16 --dry-run
PYTHONPATH=. python -m frontier.profiling.attention.dataset_tools linear-token-grid --max-tokens 16384 --exclude 4000 | wc -w   # 386
bash profiling_knowledge/scripts/profile_qwen3_mi355x.sh --stage attention --dry-run
bash profiling_knowledge/scripts/profile_qwen3_mi355x.sh --stage linear_op --dry-run
bash profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh --stage smoke --dry-run
# cluster (after implement-loop converges):
bash profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh --stage smoke
bash profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh --stage attention
bash profiling_knowledge/scripts/slurm/submit_qwen3_mi355x_profiling.sh --stage linear_op
bash profiling_knowledge/scripts/slurm/ingest_qwen3_mi355x_results.sh
PYTHONPATH=. python -m frontier.profiling.attention.dataset_tools validate \
  --dataset-dir data/profiling/compute/mi355x/qwen3-a3b-30b-moe \
  --expectation profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/expectations_qwen3_mi355x.json \
  --gpu-total-bytes "$(grep -m1 -oP 'gpu_total_bytes=\K[0-9]+' logs/qwen3_mi355x/smoke.log)" \
  --cell-logs-dir logs/qwen3_mi355x
```

## Risks / Residual Uncertainty

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| AITER memory-access fault with `num_gpus=8` (template observed it with a resident sglang server) | med | smoke stage runs 8 GPUs on an exclusive node; full run defaults to 4 (`AITER_NUM_GPUS`), per-cell split limits loss to one cell |
| Profiler has no checkpointing; a crashed cell loses ~12.5k points | med | two independent cells; `profile_cell` skips finished cells on rerun |
| Largest decode / true-mixed combos dropped by the memory filter at TP1/TP2 (96/48 KB per token) | high (expected) | validator reads the profiler's filter counts and lists dropped combos; documented coverage gap, not a failure |
| Image drift: template default `v0.5.9` absent on nodes; `v0.5.11-rocm700-mi35x` verified today (signatures match) | low | driver pins `v0.5.11`; smoke stage re-checks `inspect.signature` before the sweep |
| Strict `head_dim` is enforced only at profiler write time and by `dataset_tools validate`; the simulator still loads legacy dense CSVs lacking `head_dim` (no load-time dense validation exists) | certain (verified) | documented in runbook + docs; a dense load-time gate is a follow-up decision (simulator code, non-goal here) |
| Qwen3 `num_tokens=4000` TP1 GPU fault | certain if included | excluded from the token grid |
| aiter JIT compile time inside the sweep | low | prewarm all 8 (TP, block) pairs single-process; `/mnt/data/aiter-cache` is created by the sbatch (`mkdir -p`) and persists per node (probed on amd-mi355x-1 only) |
| NFS (`/opt/shared`) write of ~20 MB CSVs and many small logs | low | accepted |
| Another job's process resident on the GPUs → `get_max_num_blocks` over-budgets (it uses total memory) | low | `--gres=gpu:8` reserves all eight GPUs; smoke stage logs `rocm-smi --showmemuse` before profiling and aborts if any GPU has >1 GB in use |
| `rsync` of the worktree excludes `.git`; commit hash must be passed explicitly | low | `FRONTIER_COMMIT` env from the VM, written to the manifest |
| Wall-clock time of a 12.5k-point AITER cell is not recorded anywhere | med | smoke stage measures per-point time; sbatch `--time=24:00:00`; manifest records actual time |

## implement-loop Handoff

**Task file**: inline below (this plan is the task file; `00_requirements_and_source_map.md` is the source map)

**Acceptance criteria for implement-loop**:
- AC-1: `DENSE_ATTENTION_FAMILY.required_profiling_feature_columns` contains `head_dim` after `n_kv_head`; all three wrapper row builders emit `head_dim`; nine fixture tests updated; new schema tests green (TASK-1).
- AC-2: `aiter_prewarm` derives 8 Qwen3 specs / warm calls (TP {1,2,4,8} × block_size {1,16}, shard shapes (32,4), (16,2), (8,1), (4,1) × head_dim 128) and reproduces the gpt-oss shard shapes for every requested TP × block_size pair; the gpt-oss sweep script delegates to it with unchanged defaults; `bash -n` passes (TASK-2).
- AC-3: `dataset_tools` implements union / validate / linear-token-grid / manifest with the D-7 checks; expectations JSON present; tests green (TASK-3).
- AC-4: Qwen3 driver dry-runs show the `SHARED_ROOT` bind mount and `WORK_DIR`/`COLLECT_DIR`/`LOG_DIR` forwarded into docker, a separate smoke work namespace, and exactly two AITER cells (block 1, 16; TP 1 2 4 8; max_seq_len 16384; true-mixed; chunked grid) and one linear_op command (386 tokens, `--is_moe`, rope fallback); no MoE stage (TASK-4).
- AC-5: sbatch/submit/ingest scripts exist with the D-4 contracts; dry-runs green (TASK-5).
- AC-6: docs updated with the head_dim enforcement points (write-time + validator; no load-time gate) and MoE deferral (TASK-6).
- AC-7: smoke job passes on the cluster (head_dim in header, AITER backend, 8 GPUs) (TASK-7).
- AC-8: both attention cells and linear_op complete; ingest produces the D-6 file set; `dataset_tools validate` PASS; `02_run_manifest.md` filled (TASK-8, TASK-9).

**Tests to write (TDD)**: as listed in "Tests Per Task"; each backed by the family spec (TASK-1), the template `prewarm_aiter` shapes (TASK-2), grid generators + contract module (TASK-3), the request's sweep axes (TASK-4), the cluster probe (TASK-5), and this plan's decisions (TASK-6).

**Verification commands**: see "Verification Commands".

**Checklist**:
- [ ] Source check completed using sources listed in this plan
- [ ] RED evidence captured
- [ ] GREEN evidence captured
- [ ] Coverage evidence reported
- [ ] Reviewer gate passed
- [ ] Codex gate passed or repo policy fallback applied
- [ ] User approved commit
