# Evidence E5 — Approved serving image: digest, build provenance, and SGLang source verification

Collected 2026-10-04 from the DriveNets cluster login node (`amd-mi355x-1`) and this workstation. Read-only: registry
API queries, single-layer blob downloads, `docker run --rm --entrypoint bash <image> -c '<cat/pip list>'` with no GPU
devices, no model, no server. No image was pulled onto any node.

## 1. Image identity (observed in a runtime artifact: the registry)

| item | value |
|---|---|
| approved tag | `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04` (registry `10.1.1.0:5050`, HTTPS, `/v2/dnis/sglang-rocm/tags/list` lists it) |
| manifest digest (`Docker-Content-Digest`, schema 2) | `sha256:d8e0eb9c3211b593fbd574ada06e03d48d956c159bb9b6c123f4d86c12d8e6dc` |
| config blob | `sha256:dece67677b2b6cfae2aa12a546fab317a1499745d4ec866c45b9bd4ea0fc0bde`, `created 2026-09-23T11:15:32Z`, amd64, 32 layers, 11.62 GB compressed |
| presence on nodes | **not present** on `amd-mi355x-1` (only `10.1.1.0:5050/dnis/sglang-rocm:20261001-7-…` sha256:93136ea3…, built with `SGLANG_REF=v0.5.20`, and `rocm/sgl-dev:v0.5.18-rocm720-mi35x-20260824`); other nodes refused this account's SSH key; root disk on node 1 at 98 % (167 GB free) so no pull was attempted |

Build-history facts from the config blob (`history[]`, non-empty layers mapped 1:1 to the manifest's 32 layers):

- Base: `rocm/dev-ubuntu-22.04:7.2.3-complete` via the vLLM ROCm base image (`vllm.rocm_base.*` labels: rocm 7.2.3, pytorch `d0c8b1f3`, triton `0f380657`, aiter `v0.1.16.post3` in the *base*; `org.opencontainers.image.source=https://github.com/vllm-project/vllm`, revision `cbb5f045…`).
- `ARG SGLANG_REF=v0.5.18` (layer 15/17): `pip install --no-deps /tmp/sglang_kernel-*.whl` then `pip install -c torch-constraint "<sglang whl>[srt_hip]"`.
- `pip install --no-deps --force-reinstall /tmp/amd_aiter-*.whl` (layer 19) + `COPY /opt/aiter/` (layer 23) + prebaked aiter JIT `.so` (layer 21).
- `SAGUARO_SOURCE_COMMIT` label is **empty** for this tag (the 20261001-7 sibling carries `9b47ded1…`).
- Image ENV (relevant): `HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `SGLANG_USE_ROCM700A=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=1`, `SGLANG_MOE_PADDING=1`, `SGLANG_SET_CPU_AFFINITY=1`, `TORCHINDUCTOR_MAX_AUTOTUNE=1`, `NCCL_MIN_NCHANNELS=112`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`, `AITER_ROOT_DIR=/opt/aiter`. **`SGLANG_USE_AITER` is not set in the image ENV** (the `rocm/sgl-dev` image sets `SGLANG_USE_AITER=1`; the DriveNets image does not).

## 2. Exact SGLang source in the approved image (established by downloading the wheel layer only)

Layer 16 (`sha256:4b3dc133cd3e…`, 14,972,773 bytes) contains exactly one file: `/tmp/sglang-0.5.18-cp312-cp312-linux_x86_64.whl`
(sha256 `3706017cd830b60144cc964015275257d672f6811e58bcf6e2dfe064b360d75c`). Unpacked locally: `sglang/_version.py` → `__version__ = '0.5.18'`,
`__commit_id__ = None`; 3,337 `.py` files.

Reference: upstream tag `v0.5.18` = commit `71de97b264b04dcd514cf904003028aefe9775c8` (GitHub API; tag object `ff4c6e64…`; PyPI release 2026-08-21).

`diff -rq` of `python/sglang` at the tag vs the wheel (excluding `.so`, `__pycache__`, `_version.py`): **24 files differ, 19 files only in the wheel**
(all 19 new files are `srt/speculative/saguaro_*.py` and `srt/sampling/sampling_distribution.py`). Modified files:
`srt/arg_groups/speculative_hook.py`, **`srt/layers/attention/aiter_backend.py`**, `srt/layers/quantization/fp4_kv_cache_quant_method.py`,
`srt/layers/sampler.py`, `srt/managers/overlap_utils.py`, `srt/managers/schedule_batch.py`, `srt/managers/scheduler.py`,
`srt/managers/scheduler_components/{batch_result_processor,invariant_checker}.py`, `srt/managers/tokenizer_manager.py`,
`srt/mem_cache/kv_cache_configurator.py`, `srt/model_executor/model_runner.py`, `srt/model_executor/model_runner_components/spec_aux_hidden_state.py`,
`srt/model_executor/pool_configurator.py`, `srt/model_executor/runner/{decode_cuda_graph_runner,eager_runner}.py`, `srt/server_args.py`,
`srt/speculative/{base_spec_worker,eagle_utils,eagle_worker_common,eagle_worker_v2,spec_info,spec_registry,standalone_worker_v2}.py`.

Every file **not** in that list is byte-identical to the tag. In particular these are identical and can be cited from the tag: `srt/models/qwen3_moe.py`,
`srt/models/qwen2_moe.py`, `srt/models/utils.py`, `srt/layers/communicator.py`, `srt/layers/layernorm.py`, `srt/layers/linear.py`,
`srt/layers/rotary_embedding/*.py`, `srt/layers/quantization/unquant.py`, `srt/layers/vocab_parallel_embedding.py`, `srt/layers/radix_attention.py`,
`srt/mem_cache/memory_pool.py`, `srt/model_executor/forward_batch_info.py`, `srt/environ.py`, `kernels/ops/kvcache/rope_cache.py`,
`kernels/ops/attention/rope.py`.

### 2.1 The `aiter_backend.py` patch (DriveNets, 139 changed lines) — what matters for operator boundaries

1. Opt-in unified prefill: `self.use_unified_prefill = get_bool_env_var("SGLANG_AITER_UNIFIED_PREFILL", "False")` (image `:264-266`); routed only when also
   `SGLANG_USE_AITER_UNIFIED_ATTN` is set (`_can_use_unified_prefill`, image `:1955-1983`). Default path unchanged.
2. **Explicit output buffer for the legacy BF16 prefill kernel** (image `:1939-1953`, used at `:2542-2544`): for `max_q_len > 1` it calls `out.zero_()` on the
   pre-allocated `forward_batch._attn_output`, or allocates `q.new_zeros(...)` when none exists. Comment: "Caller-owned output storage avoids reproduced
   causal-prefill corruption on the tested ROCm AITER kernel for multi-token eager BF16 queries." ⇒ **one extra fill kernel precedes `mha_batch_prefill_func`
   inside the attention-core scope on this image** for every multi-token extend. The tag version passes `_attn_output` without zeroing.
3. Graph page-table transform `elif self.page_size > 1:` → `else:` (image `:1633`): the `_transform_table_1_to_real` is now applied for page size 1 too in the
   decode-graph metadata path (metadata-side only; no kernel-count change inside the attention scope).

### 2.2 aiter in the approved image

Layer 18 = `amd_aiter-0.1.1.dev1+gb4d9154d1.d20260923-cp312-cp312-linux_x86_64.whl` (`aiter/_version.py` → `0.1.1.dev1+gb4d9154d1.d20260923`), i.e. aiter commit
`b4d9154d1` built 2026-09-23 (the base image's `v0.1.16.post3` is overridden by `--force-reinstall`). Entry points present in that wheel: `aiter/ops/mha.py`
`mha_batch_prefill_func` (`:4016`), `aiter/ops/attention.py` `paged_attention_ragged` (`:728`), `aiter/ops/rmsnorm.py` `rmsnorm2d_fwd` (`:427`) /
`rmsnorm2d_fwd_with_add` (`:439`), `aiter/ops/rope.py` `rope_fwd*`, `aiter/tuned_gemm.py` `TunedGemm.mm` (`:608,632`) with `libtype ∈ {hipblaslt, asm, …}`
chosen by a tuned table. Dataset A's attention rows used aiter `d9e5ef7ce` (v0.5.17 image) and A-linear used aiter `a6bb49937` + vLLM 0.9.2 kernels
(v0.5.11 image): three different aiter commits across A-linear, A-attention and the serving image.

## 3. Default/effective flags that gate the operator paths (from the identical-to-tag files)

| gate | source (tag `71de97b2`, identical in wheel) | value |
|---|---|---|
| default attention backend, MHA on HIP | `srt/server_args.py:5917-5918` `elif is_hip(): return "aiter"` | `aiter` unless `--attention-backend` given |
| `SGLANG_USE_AITER` | `srt/environ.py:746` `EnvBool(False)`; consumers `layers/layernorm.py:53`, `quantization/unquant.py:65`, `rotary_embedding/base.py:36`, `mem_cache/memory_pool.py:114` all `get_bool_env_var("SGLANG_USE_AITER") and _is_hip` | **off by default; not set by the image** |
| `--enable-aiter-allreduce-fusion` | `server_args.py:2051-2055` default `False`; HIP auto-enable lines are commented out (`:5480`, `:5577`) | off unless passed |
| `--enable-fused-qk-norm-rope` | `server_args.py:1976-1980` default `False`; also requires `_is_cuda` (`qwen3_moe.py:522-532`) | never on HIP |
| fused RoPE + KV write | `models/utils.py:284-305`: `(_is_cuda and bf16 …) or (_is_hip and not prefill-CP and no dcp mask)` | **always on HIP** (no flag) |
| `SGLANG_USE_AITER_UNIFIED_ATTN` | `environ.py:761` `EnvBool(False)` | off |
| `SGLANG_AITER_KV_CACHE_LAYOUT` | `environ.py:793` `EnvStr("nhd")` | nhd |
| `USE_ROCM_AITER_ROPE_BACKEND` | `environ.py:796` `EnvStr("0")` | 0 |

## 4. DriveNets launch recipes found on the cluster (no Qwen3-30B-A3B recipe exists)

`/opt/shared/bp-inferencex/benchmarks/single_node/{agentic,fixed_seq_len}/*.sh` (`~/amd-playground/...` paths cited by the earlier plan do not exist for this account).
Flag census over MI355X/MI325X SGLang scripts: `--attention-backend aiter` + `SGLANG_USE_AITER=1` in the DSR1 and qwen3.5-FP4/FP8(mi325x) MTP scripts;
`--attention-backend triton` + `--enable-aiter-allreduce-fusion` and **no** `SGLANG_USE_AITER` in `agentic/qwen3.5_fp8_mi355x.sh`,
`agentic/qwen3.5_fp8_mi355x_sglang.sh`, `fixed_seq_len/qwen3.5_bf16_mi355x.sh`. `--disable-radix-cache` appears only in some scripts. So the closest
DriveNets recipes disagree on the attention backend and on AITER kernels; none is a Qwen3-30B-A3B recipe. See §5 of the report.

## 5. vLLM is absent from the DriveNets image lineage (observed on the sibling tag; inferred for the approved tag)

`docker run --rm --entrypoint bash 10.1.1.0:5050/dnis/sglang-rocm:20261001-7-… -c 'pip show vllm; python3 -c "import vllm"'` →
`WARNING: Package(s) not found: vllm` / `ModuleNotFoundError: No module named 'vllm'`; `pip list` shows `amd-aiter 0.1.1.dev1+gb4d9154d1.d20261001`,
`sglang 0.5.20`, `sglang-kernel 0.4.7`. The approved tag installs the same `[srt_hip]` extra, whose `Requires-Dist` (wheel METADATA lines 89-94) are
`compressed-tensors`, `petit_kernel`, `sglang[runtime_common]`, `torch`, `wave-lang` — no vLLM — and the same base-image lineage. Consequence (from
`srt/layers/layernorm.py:118-125, 665-674`): when `SGLANG_USE_AITER` is unset, `from vllm._custom_ops import fused_add_rms_norm, rms_norm` raises
ImportError, `_has_vllm_rms_norm=False`, and every `RMSNorm` (input/post-attention norms **and** q_norm/k_norm) runs `forward_native` (several small
PyTorch kernels), while unquantized GEMMs run `F.linear` (`unquant.py:265`). With `SGLANG_USE_AITER=1` they run aiter `rmsnorm2d_fwd[_with_add]`
and `aiter.tuned_gemm.tgemm.mm`. This fork is decided only by the launch environment, not by the image.

## 6. Two DriveNets MI355X recipe families (verbatim flag excerpts)

`fixed_seq_len/qwen3.5_bf16_mi355x.sh:34-48`: `--attention-backend triton … --enable-aiter-allreduce-fusion --cuda-graph-max-bs $CONC --disable-radix-cache
--max-prefill-tokens $MAX_PREFILL_TOKENS … --mem-fraction-static 0.8`; no `SGLANG_USE_AITER` export.
`agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh:72-110`: `export SGLANG_USE_AITER=1; export SGLANG_USE_AITER_UNIFIED_ATTN=1; export AITER_FLYDSL_FORCE=1 …
--attention-backend aiter --page-size 16 --cuda-graph-max-bs … --max-prefill-tokens 32768 --chunked-prefill-size 32768 --speculative-algorithm EAGLE …`.
Neither is Qwen3-30B-A3B; both are Qwen3.5 (a hybrid/MTP model). They differ on attention backend, page size, AITER kernels, radix cache, and speculation.
