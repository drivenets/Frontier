<!-- Evidence E4 — Operator-partition provenance from git history of Frontier-upstream / Frontier-drivenets / qwen3 worktree (subagent report). Generated 2026-10-04 by the operator-boundary investigation session. Treat as evidence to verify, not authority. -->

# Provenance of Frontier's operator partition (attention block) and the AMD/SGLang changes

All findings are from read-only git on the three checkouts. Nothing was modified. Where I quote text it is verbatim from the file/commit named; everything else is marked "inferred".

## 0. Timeline (boundary-relevant commits only)

| Commit | Date | Author | Repo | What changed w.r.t. operator boundaries |
|---|---|---|---|---|
| 6897819 | 2026-06-08 | fwyc0573 | upstream (first commit) | Entire partition already present: `linear_op_impl.py` has `attn_pre_proj`, `attn_rope`, `attn_post_proj`, `input_layernorm`, `post_attention_layernorm`, `emb`, `add`, `mlp_up_proj`, `mlp_act`, `mlp_down_proj`; `attention/backends/flashinfer_attention_wrapper.py` has `attn_kv_cache_save`, `attn_prefill`, `attn_decode`. `use_qk_norm` handling (q/k norm inside `attn_pre_proj`) already present. Attention backends carry "Adapted from sarathi-serve-vidur" headers. |
| fe70f8f | 2026-06-29 | YICHENGFENG | upstream | Introduces `frontier/attention/` (families, ops, profiling_mapping, trace_mapping); adds MLA family (`attn_mla_*` scopes) alongside dense; no change to dense scope names. |
| 639bbc2 / 9a5738b / 9d059ed / 42ccae8 | 2026-07-01 | YICHENGFENG | upstream | Move KV-head/head-size formulas, `kv_factor`, and required profiling feature columns onto `AttentionFamilySpec`; behaviour-preserving. |
| 3f750ba | 2026-07-04 | YICHENGFENG | upstream | "Lift Frontier timing onto operator families": introduces `frontier/operators/` registry (FFN_FAMILY, MEMORY_FAMILY, MOE_FAMILY, …). Predictor-side `add` is split into `add_attn_residual` / `add_ffn_residual` (both with `profiling_key="add"`, so the profiler scope name is unchanged); `emb` declared `predictor_target=False`; attention-module selection moved behind `build_linear_op_attention_module`. Scope names in the profiler unchanged. |
| 0af310e | 2026-07-24 | YICHENGFENG | upstream | PD-AF release state; touches predictor/execution_time but no scope renames. |
| c47fe32 | 2026-08-21 | YICHENGFENG | upstream | "fix: align predictor attention scope contract": adds `ExecutionTime.get_single_layer_attention_scope_time()` ("attention compute and its tensor-parallel all-reduce") and uses it for post-attention time. Predictor-side only. |
| 1128e9e | 2026-08-22 | YICHENGFENG | upstream | profiling_plan attention ops now derived from the operator registry (no name change). |
| 10d6340 / 583a618 / e7fa3a3 / b8149c3 / 3673e42 / a8d75b8 / c4ac39a / 8cc267e | 2026-08-22..23 | YICHENGFENG | upstream | Attention profiler *grid/envelope* fixes (max_seq_len vs max_model_len, mixed-batch axes, memory bounds). No scope changes. |
| 9e9fa94 | 2026-08-10 | YICHENGFENG | upstream | "update readme doc" — merge-base of DriveNets fork with upstream. |
| 3d1cabd | 2026-08-09 | nuzan | DriveNets fork | "Add AMD MI355X support…": adds `torch_sdpa_attention_wrapper.py` (472 lines, new `AttentionBackend.TORCH_SDPA`), MoE kernel changes (`moe_impl.py`, `moe_vllm_kernel.py`), MI355X SKUs. Emits the same attention scopes as FlashInfer. |
| d613032 | 2026-08-10 | nuzan | DriveNets fork | Merge of upstream/main (= 9e9fa94). |
| 645c98a | 2026-08-26 | "DriveNets" (`dn@amd-mi355x-1`) | DriveNets fork | Commit message is literally "."; adds `aiter_attention_wrapper.py` (451 lines), `AttentionBackend.AITER`, a memory-budget NOTE in `profiling/utils/__init__.py`, gpt-oss sweep CSVs. |
| deef568 | 2026-09-14 | Shaked Matar | qwen3 worktree | `head_dim` added to `DENSE_ATTENTION_FAMILY.required_profiling_feature_columns`; ACTIVE_STEPS=50; per-sample recording. |
| c7d49c3 / 74eee7c | 2026-09-16/17 | Shaked Matar | qwen3 worktree | GC-induced `attn_pre_proj` spike root-caused; gc.disable() around timed loop; `spike_diag.py`. |
| d4469be | 2026-09-17 | Shaked Matar | qwen3 worktree | Two-column timing (legacy host-bound + GPU-bound `time_stats.*`). Scope names unchanged. |
| 9dac93b | 2026-09-17 | Shaked Matar | qwen3 worktree | Shader-clock probe columns; "contract draft". |
| 4b8fce3 | 2026-09-22 | Shaked Matar | qwen3 worktree | `attn_rope` fix: the torch fallback "was never RoPE"; now uses vLLM fused `rotary_embedding` op ("the same kernel family SGLang launches on this image"); `attn_rope_impl` column. |
| 3a8eb55 / 88da6a4 | 2026-09-23 | Shaked Matar | qwen3 worktree | Backlog kind + settle steps (TimerStatsStore pause/resume), ROCclr batch-flush barrier workaround. |

Note: `git merge-base c68096c 284130b` = c68096c, i.e. the gptoss120b branch tip is an ancestor of the qwen3 branch. `a24ceda` (upstream HEAD) is not an ancestor of either fork branch.

## 1. Vidur lineage

### Documented intent (verbatim)

- `/home/dn/Frontier-upstream/README.md:178` — "Frontier is mainly built on top of Vidur. The following great systems have been referenced or adapted as runtime backends. We sincerely thank all the developers for their contributions to the community!"
- `/home/dn/Frontier-upstream/README.md:180` — "- [**Vidur**](https://github.com/microsoft/vidur)"
- `/home/dn/Frontier-upstream/AGENTS.md:80` — "  - Vidur (sklearn-based) backend trained on profiling data" (this is about the CC backend, not the operator partition)
- `/home/dn/Frontier-upstream/AGENTS.md:128`, `:566` — list `vidur` as a CC-backend type only.
- `frontier/profiling/attention/backends/base_attention_wrapper.py:1-2` — "# Copyright 2023 The Sarathi team. / # Adapted from sarathi-serve-vidur/sarathi/model_executor/attention/base_attention_wrapper.py" (same header in `flashinfer_attention_wrapper.py:2`, `no_op_attention_wrapper.py:2`).
- `frontier/profiling/common/cuda_timer.py:14` — `layer_id: int = 0,  # we don't care about layer id, it is just for compatibility with sarathi cudatimer`; `:20-22` — "# beautify the names we get from vllm … self.name = f"vidur_{name}"`.
- `frontier/profiling/utils/record_function_tracer.py:40-42` — "Run a few tiny CUDA kernels outside any vidur_* scope so the measured operation scopes remain strict."
- `frontier/profiling/example/migration_records/MIGRATION_CHANGES.md:12` — "**Objective**: Remove all Sarathi-Serve-Vidur dependencies to make Frontier completely independent"; `DEPENDENCY_ANALYSIS_REPORT.md:17` — "The `frontier/profiling/` module currently has **15 distinct import statements** from `sarathi-serve-vidur` across **9 Python files**".
- `frontier/profiling/moe/README.md:39` — "…which is used by Vidur's execution time predictor…"; `frontier/training/__init__.py:2` — "Training module for Vidur execution time predictors."

No sentence in README/AGENTS/docs explains the *operator partition rationale* in terms of Vidur. The only partition rationale text found anywhere is `frontier/profiling/linear_op/README.md:14-31` (quoted in section 4d).

### First commit check (6897819, 2026-06-08, fwyc0573)
`frontier/profiling/linear_op/linear_op_impl.py` and `frontier/profiling/attention/` (incl. `backends/flashinfer_attention_wrapper.py`) both exist in 6897819. Occurrence counts in `6897819:linear_op_impl.py`: attn_pre_proj 13, attn_rope 9, attn_post_proj 3, input_layernorm 15, post_attention_layernorm 13, emb 14, add 11, mlp_up_proj 2, mlp_act 3, mlp_down_proj 1; attn_kv_cache_save / attn_prefill / attn_decode 0 there but present in `6897819:frontier/profiling/common/constants.py:21,23,24` (`OperationMetrics.ATTN_KV_CACHE_SAVE/ATTN_PREFILL/ATTN_DECODE`) and used in `6897819:flashinfer_attention_wrapper.py:379,400,410`. So all 13 names are in the first commit. `git log --follow` shows no history before 6897819 (the repo was published as a squashed release).

Inferred: the names `attn_pre_proj/attn_post_proj/attn_rope/attn_kv_cache_save/attn_prefill/attn_decode/mlp_up_proj/mlp_act/mlp_down_proj/add/input_layernorm/post_attention_layernorm/emb` and the `vidur_` record_function prefix match Vidur's sarathi `OperationMetrics` enum (my recollection of microsoft/vidur; I could not verify against the Vidur repo in this environment — flagged).

## 2. Upstream history per path

- `frontier/profiling/linear_op/linear_op_impl.py`: 6897819 → 3f750ba only. 3f750ba replaced `"add" in enabled_ops` with `memory_operator_enabled(enabled_ops,"add_attn_residual") or …"add_ffn_residual"` and moved attention-class selection into `build_linear_op_attention_module`; the `CudaTimer("attn_pre_proj")`, `CudaTimer("attn_rope")`, `linear_metric_name="attn_post_proj"` scopes are untouched.
- `frontier/profiling/attention/`: 6897819, fe70f8f (MLA + `vllm_mla_profile_importer.py`), 9ffad9c, 639bbc2, 9d059ed, 42ccae8, 3f750ba, then eight 2026-08-22/23 envelope fixes. None rename dense scopes.
- `frontier/attention/`: created fe70f8f (2026-06-29); 9a5738b/639bbc2/331b6b0/9d059ed/42ccae8 (2026-07-01); 3f750ba, a48cd92 (07-04); 0af310e (07-24).
- `sklearn_execution_time_predictor.py`: 6897819, fe70f8f, 9ffad9c, 3f750ba, 0af310e, then ~25 Aug-2026 fixes (MoE EP, exact rows, attention finite queries, c47fe32 scope contract).
- `frontier/entities/execution_time.py`: 6897819, fe70f8f, 3f750ba (+690/-… op_times), 0af310e, 8cc8331, b8bf55f, c47fe32.

Quoted commit messages:
- 3f750ba: "Introduce operator specs, model architecture capabilities, target-embedded MTP registry support, metrics CapabilityContext, and parity tooling so profiling, predictors, metrics, and E2E validation share explicit operator-family contracts instead of scattered model-specific branches. Constraint: Preserve existing simulator/profiling numeric behavior for co-location and sequential PDD…"
- fe70f8f: "Migrate attention-family modeling, MLA profiling/prediction support, equivalence harnesses… Rejected: Fallback dense-op mapping for attn_mla_* | it would hide missing metadata and ledger contracts."
- 639bbc2: "Move dense and latent-MLA runtime KV-head/head-size formulas onto AttentionFamilySpec so model configs and profiling wrappers dispatch through the family binding instead of a second raw use_mla branch."
- c47fe32 / 1128e9e / 7cfe5a3 / c4ac39a / a8d75b8: one-line subjects only (no body).

No upstream commit adds `attn_kv_cache_save` or `use_qk_norm` — both predate the public history. Mixed-batch (`README_MIXED_BATCH.md`) and chunked-prefill (`profile_attention_chunked_prefill.sh`) are also already in 6897819; `README_MIXED_BATCH.md` modification history dates its own first entry 2025-11-09 (pre-publication).

## 3. DriveNets fork commits not in upstream (base 9e9fa94)

`git log 9e9fa94..c68096c -- frontier/{profiling,attention,execution_time_predictor,training,entities}`: 3d1cabd, d613032, 645c98a. `9e9fa94..284130b` adds the eight Shaked Matar commits listed in the timeline. Details:

- **3d1cabd** (2026-08-09, nuzan, `nuzan@drivenets.com`): "Port MI355X device/node SKUs, profiling data, and attention/MoE kernel support from FrontierBase (9f1d4617)." Files: `attention/backends/__init__.py` (+14, `TORCH_SDPA` enum), `attention/backends/torch_sdpa_attention_wrapper.py` (+472), `moe/moe_impl.py` (+73), `moe/moe_vllm_kernel.py` (+31). Docstring `torch_sdpa_attention_wrapper.py:5-10`: "…FlashInfer has no ROCm build, so it raises ImportError on AMD hardware… That leaves no way to profile attention on ROCm (e.g. MI355X / gfx950)." Line 436 region: "Emits the same scopes as the FlashInfer backend". Note: "FrontierBase 9f1d4617" and "90a22702" are not objects in this repo — provenance of that earlier work is not recoverable here (flagged).
- **645c98a** (2026-08-26, author "DriveNets <dn@amd-mi355x-1>", message "."): adds `aiter_attention_wrapper.py`, `AITER` backend enum, and a NOTE in `profiling/utils/__init__.py` about `mem_get_info()` budgeting against total memory ("an sglang server holding 253 of 288 GB here").
- **qk_norm**: `git log -S'qk_norm' 9e9fa94..284130b -- frontier/` is empty; `git diff 9e9fa94 284130b -- linear_op_impl.py` has no qk_norm/rope/scope lines. The fork never touched `linear_op_impl.py`.
- **GPU-bound timing**: d4469be (two-column), 9dac93b (clock probe), 3a8eb55/88da6a4 (backlog + settle) — all in `linear_op_wrapper.py`, `cuda_timer.py` (+5: pause support), `timer_stats_store.py`. Scope names unchanged.
- **rope fix 4b8fce3**: `common/layers/rotary_embedding.py` (+100/-…), `linear_op_wrapper.py`, `record_function_tracer.py` (keep_trace flag).

## 4. Specific determinations

**(a) `use_qk_norm` and placement inside `attn_pre_proj`.** Present in the first upstream commit; no later commit touches it. Documented intent (verbatim, `/home/dn/Frontier-upstream/frontier/profiling/linear_op/linear_op_impl.py`, HEAD line numbers; identical text at 6897819 lines 419-456/514):
- `:424` "# QK-norm support (for Qwen3, Gemma3, OLMo2, etc.)"
- `:430-431` "# When QK-norm is enabled, we use an external timer to wrap / # QKV projection + QK-norm together (matching vLLM's attn_pre_proj scope)"
- `:458-459` "# Keep attn_pre_proj boundary at the attention scope so the timed / # region can include qkv split, matching vLLM's source contract."
- `:502` "# QK-norm enabled: wrap QKV projection + QK-norm in single timer"
- `:518` "# Apply QK-norm (inside attn_pre_proj scope to match vLLM)"
So yes, deliberate; the stated rationale is matching vLLM's scope, not any token/sequence-dependence argument. The Step3 variant (`:285-336`) additionally has a nested `attn_pre_proj_q_norm` timer inside `attn_pre_proj`. `use_qk_norm` is also emitted as a CSV column (`linear_op_wrapper.py:190`) and enforced by `training/linear_op_trainer.py:368-397` and `training/attention_trainer.py:177-186`.

**(b) `attn_kv_cache_save` as a separate scope.** Already separate in 6897819 (`constants.py:21`; `flashinfer_attention_wrapper.py:379`). No commit message or comment gives a rationale. The only descriptive text: `frontier/attention/families.py:54-60` (HEAD) declares it `role=CACHE_WRITE, phases=_ALL_PHASES, resource_class=ResourceClass.MEMORY` while `attn_prefill`/`attn_decode` are `ResourceClass.COMP`; `attention_tp_policy.py:7-21` (6897819) groups it under `ATTENTION_NON_LINEAR_OPS` with prefill/decode vs `ATTENTION_LINEAR_OPS = {attn_pre_proj, attn_post_proj, attn_rope}`. Inferred: inherited from Vidur/sarathi, where the KV write (`reshape_and_cache`) is timed separately from the attention kernel.

**(c) AITER backend origin.** Local to DriveNets. `git log --all --follow -- frontier/profiling/attention/backends/aiter_attention_wrapper.py` → only 645c98a (2026-08-26, "DriveNets <dn@amd-mi355x-1>", message "."). Upstream HEAD a24ceda has no file matching `aiter` (`git ls-tree -r HEAD | grep -i aiter` → none). Its docstring (`aiter_attention_wrapper.py:11-20`) ties it to SGLang: "Real sglang serving on MI355X does not run SDPA — it dispatches AMD's aiter kernels. This backend calls the same two entry points sglang's own aiter_backend.py uses for dense/GQA (non-MLA) models… prefill aiter.ops.mha.mha_batch_prefill_func (aiter_backend.py L1524); decode aiter.ops.attention.paged_attention_ragged (aiter_backend.py L1633)". It emits the same scopes (`:394 ATTN_KV_CACHE_SAVE`, `:403 ATTN_PREFILL`, `:423 ATTN_DECODE`, plus input/output reshape). The author identity is a shared machine account, so the human author is not determinable from git (flagged).

**(d) Why operators are grouped.** Documented intent is thin:
- `frontier/profiling/linear_op/README.md:14` — "Linear operations are characterized by having **linear complexity with respect to sequence length**."
- `:16-31` — "### Rationale for Renaming from MLP to linear_op … 1. **Broader Scope**: The module profiles not just MLP layers, but all operations with linear complexity: MLP layers: mlp_up_proj, mlp_down_proj, mlp_act; Normalization: input_layernorm, post_attention_layernorm; Attention projections: attn_pre_proj, attn_post_proj, attn_rope; Residual connections: add. 2. **Better Categorization**: Aligns with the three-category model structure: attn: Attention operations (prefill, decode, KV cache); moe: …; linear_op: Linear operations (this module). 3. **Extensibility**…"
- `shared_prediction_model_manager.py` (6897819 `:66-79`, HEAD `:159`): "Input → [input_layernorm] → [attn_pre_proj → attn_rope → attn_prefill/decode → attn_kv_cache_save → attn_post_proj] → [add] → [post_attention_layernorm] → [mlp_up_proj → mlp_act → mlp_down_proj] or [moe_…] → [add] → Output"; `:1033-1036` (6897819) "Pre-attention normalization (from linear_op.csv): input_layernorm; Attention projections (from linear_op.csv): attn_pre_proj, attn_post_proj, attn_rope; Attention core operations (from attention.csv): attn_kv_cache_save, attn_prefill, attn_decode".
- 3f750ba's registry adds machine-readable grouping (`frontier/operators/families.py:249-306` MEMORY_FAMILY = input_layernorm, post_attention_layernorm, add_attn_residual, add_ffn_residual, emb, all `TensorParallelMode.REPLICATED`, `ResourceClass.MEMORY`; FFN_FAMILY `:31-70` all `FFN_TP`, `COMP`) and `spec.py:47-52` `ProjectionOwnership {OUTSIDE_ATTENTION, INSIDE_ATTENTION_PHYSICAL_SCOPE, NOT_PROJECTION}`.
No commit message or doc in either repo states a token-level vs sequence-level dependence argument explicitly; the closest is the "linear complexity with respect to sequence length" sentence. The two-CSV split (linear_op keyed on num_tokens/TP; attention keyed on batch_size/prefill_chunk_size/kv_cache_size/is_prefill — `profiling/README.md:621-636`) is the operational expression of it (inferred).

## 5. feat/gptoss120b-mi355 vs qwen3 worktree

`git log 284130b..c68096c -- frontier/profiling frontier/attention frontier/execution_time_predictor` is **empty**; c68096c is the merge-base of the two tips. The `--stat` diff (10 files, +93/-651) is entirely the qwen3 branch's additions (two-column timing, rope fix, spike_diag.py, head_dim column, settle/backlog) being absent on the gptoss branch. The gptoss branch has no profiler/attention change the qwen3 worktree lacks.

## Could not determine / flags
- Vidur name match asserted from memory; microsoft/vidur not available locally for a line-level diff.
- "FrontierBase (9f1d4617)" / "90a22702" referenced by 3d1cabd are not objects in these repos.
- 645c98a's human author (committed as "DriveNets <dn@amd-mi355x-1>" with message ".").
- Upstream history is a squashed public release starting 2026-06-08; `README_MIXED_BATCH.md` and `linear_op/README.md` carry internal changelog dates back to 2025-11/12, so the real introduction dates of the partition are pre-publication and unrecoverable here.
