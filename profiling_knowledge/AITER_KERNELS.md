# AITER Kernels: What Exists, Where, and the Gap for Dense/GQA Models

`aiter` is AMD's own optimized kernel library. Real sglang deployments on MI355X use it
directly — every sglang launch command captured in `tools/inference_bench/` explicitly passes
`--moe-runner-backend aiter`. Frontier's own profiling, by contrast, has been using
`TORCH_SDPA` — a portable, correctness-first reference backend, explicitly documented (in both
[MI355X_ROCM_COOKBOOK.md](MI355X_ROCM_COOKBOOK.md) and `frontier_cli_translator.py`'s own
docstring) as **not peak-tuned**. That gap is a real, plausible contributor to any large
real-vs-simulated mismatch you see in a comparison report.

## What exists today, and exactly where

> **Updated 2026-08-11 — this section's "no MLA backend on server1" claim is now out of date for
> `server1:~/frontier_work/drivenetsfrontier`.** Both MLA wrappers
> (`torch_sdpa_mla_attention_wrapper.py`, `aiter_mla_attention_wrapper.py`) were ported into that
> checkout from `server3`, along with the `attention_wrapper.py` / `main.py` plumbing they need.
> `TORCH_SDPA_MLA` is verified working there end to end, including true-mixed batches. `AITER`
> dispatches correctly but **cannot execute on that host's torch** — see
> [Prebuilt AITER kernels vs. host torch](#prebuilt-aiter-kernels-vs-host-torch-the-real-blocker-today)
> below, which is a stack-level blocker, not a checkout-drift one.

Frontier's own `frontier.profiling.attention.main --attention_backend` choices are
`{FLASHINFER, NO_OP, TORCH_SDPA, FLASHINFER_MLA}` — **no AITER option** — on every checkout we
have direct control over: `/home/dn/FrontierBase`, `/home/dn/driventes-frontier`, and
`server1` (`amd-mi355x-1`, both `~/frontier_work/drivenetsfrontier` and
`~/frontier_work/Frontier`).

`server3` (`amd-mi355x-3`) and `server8` (`amd-mi355x-8`) are different: both have
`~/frontier-work/Frontier/frontier/profiling/attention/backends/aiter_mla_attention_wrapper.py`,
and their `AttentionBackend` enum includes a real `AITER` value:

```python
class AttentionBackend(Enum):
    FLASHINFER = "FLASHINFER"
    NO_OP = "NO_OP"
    TORCH_SDPA = "TORCH_SDPA"
    TORCH_SDPA_MLA = "TORCH_SDPA_MLA"
    AITER = "AITER"
```

`get_attention_wrapper()` dispatches `AttentionBackend.AITER` straight to
`AiterMlaAttentionWrapper` — **there is no generic AITER path, "AITER" in this branch means
"AITER for MLA," full stop.**

## What the MLA wrapper actually does

This is real, careful, already-validated work, not a stub. Reading its own docstring:

- It calls AMD's actual production `aiter` MLA kernels — the same ones real sglang serving
  uses via `--attention-backend aiter` — instead of the portable
  `TorchSdpaMlaAttentionWrapper` reference implementation.
- It was built by reading real sglang's own `aiter_backend.py` from the exact
  `lmsysorg/sglang:v0.5.11-rocm700-mi35x` image used for the real DeepSeek-R1 benchmarks it's
  meant to match — including catching and correcting a wrong initial assumption about
  decode-side query absorption (`q` arrives at `forward_decode` already absorbed into
  rank-space; `aiter`'s `mla_decode_fwd` does not do that absorption itself).
- Prefill dispatches through a **persistent fp8 kernel**
  (`mla_fp8_prefill_attn` → `aiter.mla_prefill_ps_asm_fwd`), not the generic
  `aiter.mla_prefill_fwd` (which asserts a shape constraint this model's real dimensions violate:
  `qk_head_dim=192 != kv_lora_rank+qk_rope_head_dim=576`).

## Prebuilt AITER kernels vs. host torch (the real blocker today)

Having the wrapper is not enough: the `aiter` Python package and its compiled kernels must also
match the torch you run against. On `server1` (`torch 2.11.0.dev20251216+rocm7.0`) this was
walked all the way down, and the result is that **no AITER attention profiling is runnable on
the host stack right now — on server1 or server3**:

| Thing tried | Where it came from | Result |
|---|---|---|
| `aiter` from `server3:/usr/local/lib/python3.10/dist-packages` | pip install, `v0.0.0` | Imports, JIT-builds fine, but has **no `get_ps_metadata_info_v1`** (only `get_mla_metadata_info_v1`) — the wrapper's prefill path needs it |
| `aiter` from `server3:~/dn_code/aiter` | source checkout | Has the ps-metadata API, but `aiter/__init__.py` swallows its own import errors, leaving `aiter.dtypes` unset → `aiter.mla` unimportable |
| `aiter` from `server3:~/gpt_oss_rebase/aiter` | the sglang-image lineage the wrapper was written against | Has **every** symbol the wrapper needs (`dtypes`, `get_ps_metadata_info_v1`, `get_ps_metadata_v1`, `mla_prefill_ps_asm_fwd`, `mla_reduce_v1`) |

With that last one in place the wrapper dispatches correctly and then dies inside the kernels
themselves:

- **Prefill**: `aiter.get_ps_metadata_v1` → `RuntimeError: set_stride is not allowed on a Tensor
  created from .data or .detach()`, raised from inside `module_ps_metadata.so`. Not an
  inference-mode artifact — it reproduces under plain eager, `no_grad`, and `inference_mode`
  alike, and equally when bypassing aiter's torch-custom-op layer and calling the raw pybind
  module directly. **The identical probe fails the same way on server3**, using server3's own
  aiter checkout and its own `~/aiter_jit_cache` — so this is not local-port damage.
- **Decode**: `module_mla_asm.so` fails to load at all —
  `undefined symbol: _ZN3c103hip28c10_hip_check_implementationEiPKcS2_ib`, i.e. it was linked
  against a different `c10` ABI than the installed torch.
- **Rebuilding from source doesn't help**: `module_ps_metadata`'s JIT build fails to compile
  `csrc/include/custom_all_reduce.cuh` under ROCm 7.2.4's clang (`no template named 'packed_t'`,
  cascading `unknown type name 'P'`).

The prebuilt `.so` files date from March and were built for the
`lmsysorg/sglang:v0.5.11-rocm700-mi35x` container's torch. Closing this means running the
profiling **inside that container** (where kernels and torch match), or rebuilding aiter against
the host torch — which is blocked on the CK/compiler mismatch above. Until one of those happens,
`TORCH_SDPA_MLA` is the only executable MLA backend, on any of these machines.

Two smaller traps found on the way, worth keeping:

- **`hipcc` discovery**: aiter resolves the compiler as `$ROCM_HOME/bin/hipcc`. On server1
  `ROCM_HOME=/opt/rocm` → `/opt/rocm-7.0.1`, a partial tree with no `bin/` at all, while the real
  toolchain (what `which hipcc` resolves to) is `/opt/rocm-7.2.4`. Every JIT build fails with a
  bare `/opt/rocm/bin/hipcc: not found` until you `export ROCM_HOME=/opt/rocm-7.2.4`
  (`profile_true_mixed_batch.sh` now does this automatically for `--attention-backend AITER`).
- **Prebuilt module cache**: aiter looks for `module_*.so` in `$AITER_JIT_DIR` (falling back to
  its own `aiter/jit/` dir when writable). server3 keeps a populated one at `~/aiter_jit_cache`;
  a mirror now lives at `server1:/home/dn/aiter_jit_cache_frontier`. Point `AITER_JIT_DIR` at a
  directory **you own** — server1's pre-existing `~/aiter_jit_cache` has a root-owned `build/`
  subdirectory, and the resulting `PermissionError` surfaces as the very confusing
  `ImportError: cannot import name 'dtypes' from 'aiter'`, because `aiter/__init__.py` swallows
  the real exception.

## The gap: nothing exists for dense/GQA models

gpt-oss (and Qwen3-30B-A3B) use plain GQA attention — no latent compression, no rank-space
absorption, none of the MLA-specific math this wrapper is built around. It cannot be pointed at
these models. **No dense/GQA AITER wrapper exists on any checkout we've found.**

Building one would mean repeating the same kind of work that produced the MLA wrapper, but for
sglang's *dense* attention dispatch path instead:

1. Read real sglang's `aiter_backend.py` for its non-MLA (GQA) forward-path dispatch — find
   which actual `aiter` kernel(s) it calls for prefill and decode on a GQA model.
2. Implement a new wrapper against Frontier's `BaseAttentionWrapper` interface, following the
   `aiter_mla_attention_wrapper.py` pattern (and cross-check against the correctness-oracle
   `TorchSdpaAttentionWrapper` the way the MLA one was validated).
3. Register it as a new backend value (or extend `AttentionBackend.AITER`'s dispatch to branch
   on attention family, dense vs. MLA, rather than hardcoding MLA).

This is real, unverified-until-run engineering work — not a config flag we forgot to flip.

## How to use what already exists (DeepSeek/MLA, today)

If you're validating DeepSeek-R1 specifically, this is ready right now on `server3`/`server8`:

```bash
python3 -m frontier.profiling.attention.main \
  ... \
  --attention_backend AITER \
  ...
```

It is **not** on `server1`, and not in either `FrontierBase` or `driventes-frontier` — sync
`aiter_mla_attention_wrapper.py` (and the updated `backends/__init__.py`) there first if you want
to run it from one of those checkouts instead. See
[INFRASTRUCTURE_MAP.md](INFRASTRUCTURE_MAP.md) for the full checkout inventory and what's missing
where.

**Packaged as a runnable script**: [`scripts/profile_deepseek_aiter_mla.sh`](scripts/profile_deepseek_aiter_mla.sh)
wraps the command above with a preflight check that fails fast with this same
guidance if `AttentionBackend.AITER`/`aiter_mla_attention_wrapper.py` isn't
present on the checkout it's run from (`--skip-aiter-preflight` bypasses it if
you've verified otherwise), plus a `--attention-backend TORCH_SDPA_MLA` mode
for the portable reference backend on checkouts that don't have AITER at all.
Unlike `profile_true_mixed_batch.sh`, its default grid is sanity-check scale,
not a validated production sweep — there's no single "final AITER command"
recorded from this session to reproduce exactly; widen it deliberately for
your real workload shape.
