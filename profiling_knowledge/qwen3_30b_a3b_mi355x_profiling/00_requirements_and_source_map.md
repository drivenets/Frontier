# Qwen3-30B-A3B on MI355X — plan-loop Stage 0–2 record

Branch: `smatar/qwen3-30b-mi355-profiling` (worktree `/home/dn/Frontier-qwen3-profiling`, based on `feat/gptoss120b-mi355` @ c68096c per the user's Stage 3 decision; the request's "off main" was superseded).
Request file: `/home/dn/qwen_profiling_plan_01.md` (2026-09-10).
Companion plan: `01_plan.md` (written after Stage 3 decisions are recorded).

## Stage 1 — Requirements validation

```text
REQUIREMENTS_VALIDATION:
  status: COMPLETE
  goal: Produce a verified implementation plan for collecting fresh MI355X profiling
        data (AITER backend, BF16, cuda_event) for a selected set of Qwen3-30B-A3B
        (`qwen3-a3b-30b-moe`) operators, with the head_dim schema fix, such that the
        resulting CSVs pass a written validation checklist and are ready for regressor
        training in the next phase.
  non_goals:
    - Training/fitting regressors (next phase).
    - Any simulator/binding/precision-path code change for this model.
    - Any other model (Qwen3-235B-A22B explicitly not chosen; request pre-selects 30B-A3B).
  constraints:
    - Repo: drivenets/Frontier fork; new branch off main named smatar/qwen3-30b-mi355-profiling.
    - Device: AMD MI355X (gfx950). Attention via AITER backend (sglang production kernels),
      not TORCH_SDPA.
    - Reuse the gpt-oss profiling scaffolding
      (profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh) as the template.
    - Attention rows are selected by EXACT-MATCH on block_size; profile at block_size ∈ {1, 16}.
    - attention_dataset_contract.py rejects a plain attention.csv lacking mixed-batch columns
      when combined/true_mixed siblings exist; output file set must stay internally consistent.
    - head_dim must become a recorded + required dense-family column before the run.
    - Sweep: TP ∈ {1,2,4,8}; contexts ≥ 16k; true-mixed batches on; chunked-prefill grid on.
    - GPU work runs on the Slurm cluster (partition XAI, nodes amd-mi355x-[1-9]) via
      `ssh cluster "sudo -u dn sbatch …"`; the VM has no GPUs (project memory rule).
    - ForgeLoop gates (plan-loop → implement-loop, Codex review) may not be skipped (CLAUDE.md).
  success_criteria:
    - Finalized operator list with include/exclude reasoning, derived from the real manifest.
    - Exact grid/settings per operator, with point counts.
    - Explicit new-vs-reused script/config inventory.
    - Output paths + schema (incl. head_dim) written down.
    - A validation checklist (row counts, grid coverage, schema, no duplicate/symlink data)
      executable against the collected CSVs before regressor hand-off.
  known_risks:
    - The AITER dense wrapper lives only on feat/gptoss120b-mi355, not on main.
    - moe_grouped_gemm has no AITER path in Frontier's MoE profiler (vLLM Triton fused_moe only).
    - Making head_dim required breaks every legacy dense attention CSV at predictor load.
    - Profiler has no checkpointing; per-cell splitting is the only crash protection.
    - Qwen3 linear_op GPU fault at num_tokens=4000, TP=1 (documented, deterministic).
  user_answers_summary: |
    Taken verbatim from the request file: model = Qwen3-30B-A3B (`qwen3-a3b-30b-moe`);
    branch off main named like smatar/qwen3-30b-mi355-profiling; task = (1) select several
    operators from the real manifest, (2) collect fresh MI355X data; out of scope = regressor
    fitting, binding/precision code, other models; selection criteria (a) TTFT/TPOT share,
    (b) extrapolation exposure, (c) portable-vs-production kernel; environment = MI355X,
    AITER backend, reuse gpt-oss sweep scaffolding; block_size {1,16}; contract-safe output
    file set; add head_dim to recorded+required columns; TP {1,2,4,8}, contexts ≥16k,
    true-mixed on, chunked-prefill grid on; deliverables 1–5 as listed in the request.
```

## Stage 2 — Reliable-source map

```text
SOURCE_MAP: ESTABLISHED
SOURCES:
  authoritative:
    - name: Request brief
      location: /home/dn/qwen_profiling_plan_01.md
      authority_rank: 1
      covers: goal, scope, constraints, sweep axes, deliverables
    - name: Operator binding
      location: frontier/operators/binding.py:52-110 ; frontier/operators/families.py
      authority_rank: 2
      covers: which operator families/operators resolve for an MoE dense-GQA model
    - name: Dense attention family spec
      location: frontier/attention/families.py:51-98 (required_profiling_feature_columns:79-91)
      authority_rank: 2
      covers: dense schema; where head_dim must be added
    - name: Attention profiler entry point
      location: frontier/profiling/attention/main.py (args 361-666; output 1912-2020; validation 870-892)
      authority_rank: 2
      covers: CLI grid flags, output files, config yaml, write-time schema validation
    - name: Attention wrapper metadata rows
      location: frontier/profiling/attention/attention_wrapper.py:388-402, 493-503, 563-573
      authority_rank: 2
      covers: the three row-builders that must record head_dim
    - name: Profiling ModelConfig
      location: frontier/profiling/common/model_config.py:286 (from_model_name), 455-463 (get_head_size), 557 (to_dict)
      authority_rank: 2
      covers: head_dim is already carried (explicit 128 for this model)
    - name: Grid generators
      location: frontier/profiling/utils/__init__.py:156-218, 242-411, 885-934
      authority_rank: 2
      covers: standard / true-mixed attention grids, num_tokens grid; point counts
    - name: AITER dense wrapper (port source)
      location: git feat/gptoss120b-mi355 — frontier/profiling/attention/backends/aiter_attention_wrapper.py, backends/__init__.py (commit 645c98a; 3 commits ahead of main)
      authority_rank: 2
      covers: production-kernel attention backend; kernel call contract
    - name: gpt-oss full sweep script (template)
      location: git feat/gptoss120b-mi355 — profiling_knowledge/scripts/profile_gptoss_attention_full_sweep.sh
      authority_rank: 2
      covers: per-cell splitting, prewarm, docker, collect step, env-var grid
    - name: Attention dataset contract
      location: frontier/execution_time_predictor/attention_dataset_contract.py
      authority_rank: 2
      covers: sibling-file rule for mixed columns
    - name: Predictor attention loader
      location: frontier/execution_time_predictor/sklearn_execution_time_predictor.py:1195-1233 ; shared_prediction_model_manager.py:1682
      authority_rank: 2
      covers: exact-match filter (n_embd, n_q_head, n_kv_head, block_size, TP); schema validation at load
    - name: Attention TP policy
      location: frontier/execution_time_predictor/attention_tp_policy.py:26-32, 54-108 ; frontier/profiling/common/model_config.py:422-454
      authority_rank: 2
      covers: TP=8 with 4 KV heads is valid (replication); attention linear ops set
    - name: linear_op profiler
      location: frontier/profiling/linear_op/main.py:224-400 ; linear_op_impl.py:425-528 (QK-norm inside attn_pre_proj)
      authority_rank: 2
      covers: linear-op CLI, --is_moe, num_tokens grid, QK-norm scope
    - name: MoE profiler
      location: frontier/profiling/moe/main.py:166-360 ; moe_wrapper.py:540-580 ; moe_vllm_kernel.py
      authority_rank: 2
      covers: MoE CLI, TP×EP, load distributions, grouped-GEMM backend = vLLM fused_moe (Triton)
    - name: MI355X collection driver
      location: examples/profiling/profile_mi355x.sh
      authority_rank: 2
      covers: linear_op/moe invocation pattern; Qwen3 num_tokens=4000 exclusion; MoE token grid
    - name: Model config
      location: data/config/models/qwen3-a3b-30b-moe.json
      authority_rank: 2
      covers: 32 q heads / 4 kv heads / head_dim 128 / 128 experts top-8 / hidden 2048 / bf16 / 48 layers
    - name: Existing tests pinning the dense schema
      location: tests/unit/test_attention_family_specs.py:765-777 ; test_attention_family_spec_data.py:25-37 ; test_attention_main_true_mixed_contract.py:130-215 ; test_profiling_dataset_metadata_contract.py
      authority_rank: 2
      covers: tests that must change with head_dim
    - name: Cluster environment (probed 2026-09-10)
      location: `ssh cluster` → Slurm partition XAI, nodes amd-mi355x-[1-9]; image lmsysorg/sglang:v0.5.11-rocm700-mi35x present on amd-mi355x-1 with torch 2.9.0a0, vllm, sglang, aiter, /opt/rocm/bin/hipconfig; /mnt/data/aiter-cache exists; /dev/kfd gid 110
      authority_rank: 2
      covers: run environment; kernel signatures verified against the wrapper's calls
  context_only:
    - name: GPTOSS_TRUE_MIXED_BATCH_PROFILING.md
      location: profiling_knowledge/GPTOSS_TRUE_MIXED_BATCH_PROFILING.md
      authority_rank: 3
      note: prior-run narrative (86-hour lesson, block_size facts); not implementation truth
    - name: AITER_KERNELS.md / MI355X_ROCM_COOKBOOK.md / INFRASTRUCTURE_MAP.md / MI355X_FOUR_MODEL_PROFILING.md
      location: profiling_knowledge/
      authority_rank: 3
      note: hard-won gotchas (CUDA_VISIBLE_DEVICES, rope fallback, sglang uses --moe-runner-backend aiter); partially stale (AITER dense wrapper now exists)
    - name: Frontier model fidelity audit
      location: amd-playground/poc-mi355x-qkv-gemm-profile/frontier_fidelity/01_frontier_architecture_fidelity_audit.md
      authority_rank: 3
      note: why this model; head_dim gap; not implementation truth
    - name: Existing MI355X Qwen3 data
      location: data/profiling/compute/mi355x/qwen3-a3b-30b-moe/*.csv (TORCH_SDPA, block 16, max_model_len 9472)
      authority_rank: 3
      note: superseded target; used only to size grids and as legacy-file case
CONFLICTS:
  - Request says "branch off main"; the required AITER dense wrapper and the gpt-oss sweep
    template exist only on feat/gptoss120b-mi355 (3 commits, 4 code files + ~240k lines of
    gpt-oss CSV). Resolved in Stage 3 by the user: "Rebase onto feat branch" — the branch is based on
    feat/gptoss120b-mi355 @ c68096c, taking wrapper, script and gpt-oss data as they are.
MISSING_AUTHORITY:
  - No AITER MoE (grouped GEMM) backend exists in Frontier's MoE profiler; sglang production
    uses --moe-runner-backend aiter. Stage 3 decision.
  - The request does not say how legacy dense CSVs (80 files across 7 devices, none with
    head_dim) should behave once head_dim is required. Stage 3 decision.
```

## Stage-one enumeration — the operator manifest Frontier actually resolves

Resolved by running `build_operator_manifest(BaseModelConfig.create_from_name("qwen3-a3b-30b-moe"))`
on this checkout (2026-09-10). Attention binding: `dense_attention/gqa`. `is_moe=True`,
`supports_share_expert()=False`, `get_head_dim()=128`, architecture profile `generic`.

| Family / variant | Operator | Role | Profiler that measures it | CSV |
|---|---|---|---|---|
| dense_attention/gqa | attn_kv_cache_save | cache_write | attention | attention*.csv |
| dense_attention/gqa | attn_prefill | prefill_kernel | attention | attention*.csv |
| dense_attention/gqa | attn_decode | decode_kernel | attention | attention*.csv |
| memory/replicated | input_layernorm | normalization | linear_op | linear_op.csv |
| memory/replicated | post_attention_layernorm | normalization | linear_op | linear_op.csv |
| memory/replicated | add_attn_residual, add_ffn_residual | residual (profiling key `add`) | linear_op | linear_op.csv |
| memory/replicated | emb | embedding (predictor_target=False) | linear_op | linear_op.csv |
| moe/routed | moe_gating_linear | projection | moe | moe.csv |
| moe/routed | moe_gating_routing_topk | reshape | moe | moe.csv |
| moe/routed | moe_shuffling | reshape | moe | moe.csv |
| moe/routed | moe_grouped_gemm | projection | moe | moe.csv |
| attention-layer linear ops (not a manifest family; predictor inputs per `attention_tp_policy.ATTENTION_LINEAR_OPS`) | attn_pre_proj (QKV GEMM **+ QK-norm** inside the same timer, `linear_op_impl.py:501-526`), attn_rope, attn_post_proj | projection / position_encoding | linear_op | linear_op.csv |

Not bound for this model: `ffn/dense` (MoE model), `share_expert` (no shared expert),
`kv_transfer` (request-level, not profiled). `comm/collective` operators are network
collectives measured by `frontier.profiling.collectives`, out of this compute-operator scope.
`lm_head_linear`/`mtp_fusion_proj` are only profiled with `--include_target_embedded_mtp`
(speculative-decoding targets) and are not part of this model's manifest.
