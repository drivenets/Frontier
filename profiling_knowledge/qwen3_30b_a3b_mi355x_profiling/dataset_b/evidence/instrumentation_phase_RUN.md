# Dataset B instrumentation phase — 2026-10-04

Scope: items 1-2 (and part of 3) of `03_COLLECTION_AGENT_PROMPT.md` — freeze the recipe inputs, implement
real source-matched instrumentation, fix the native-trace blockers it exposed. **No model server was
started and no real pilot request was sent in this phase.** That is the next phase.

## 1. Model/recipe inputs verified (cluster, CPU-only bounded jobs, node `amd-mi355x-3`)

- Model: `/mnt/data/data/hai/qwen-blockwise-fp8-ab-20260810/models/Qwen3-30B-A3B` — all 16/16
  `model-*.safetensors` shards present, header keys match `model.safetensors.index.json`'s weight map
  exactly (checked shard 1 and 16), **every shard is pure BF16** (union of dtypes across all 16 shards:
  `{BF16}`), not the neighboring FP8 checkpoint. Tokenizer complete (`tokenizer.json` BPE,
  `tokenizer_config.json` w/ chat template, `vocab.json`, `merges.txt`, `generation_config.json`). HF hub
  revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (from `.cache/huggingface/download/*.metadata`).
  File hashes recorded in the shell transcript of this session (config.json, tokenizer*, index.json).
  This corrects the handoff's stale historical path (`/mnt/data/data/hai/hai2/models/Qwen3-30B-A3B`,
  confirmed absent on this node).
- Approved image still verified present/correct on `amd-mi355x-3`: `sha256:dece6767...` (job 21725,
  prior session).

## 2. rocprofv3 ROCTX marker mechanism — real bug found and fixed (jobs 21736, 21737)

See `evidence/rocprofv3-format-probe2-21737/RUN.md` for full detail. Summary:

- **Bug**: `marker_runtime.MarkerBuffer` dlopened `libroctx64.so`; rocprofv3 1.1.0 never instruments that
  library (`RTLD_LOCAL` and `RTLD_GLOBAL` both tried; zero marker events, no `MARKER (ROCTx) ...
  initialized` log line). Fixed: dlopen `librocprofiler-sdk-roctx.so` instead (ABI-identical symbols);
  confirmed this is the library rocprofv3 hooks (`MARKER (ROCTx) version 1.1.0 initialized` + a real
  `*_marker_api_trace.csv`).
- **Schema discovered**: `*_marker_api_trace.csv` has the same 7 columns as `hip_api_trace.csv`
  (`Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp`); one row IS one
  complete push/pop range (`Domain=MARKER_CORE_RANGE_API`, label text verbatim in `Function`) — not a
  push/pop event stream. `b_traces.pair_markers` (stack-based pairing) was written for a different,
  apparently-hypothetical dialect; added `b_traces.rocprof_marker_csv` for the real one and wired it into
  `normalize_b.py` as an additive `source["marker_csv"]` key (kept `marker_events`/`pair_markers` as-is,
  unused by real rocprofv3 CSV output but not proven dead for every tracer).
- **Attribution chain verified against real kernels**: parsed all three real CSVs with Python's `csv`
  module (naive `awk -F','` breaks on kernel names containing literal commas). Of 6 total
  `hipLaunchKernel`/`hipExtModuleLaunchKernel` calls in the trace, exactly the 3 real `a @ b` GEMM
  launches (not the 3 earlier `torch.randn` launches) have host timestamps fully inside the marker
  range; `Correlation_Id` joins hip_api → kernel_trace exactly as `b_traces.rocprof_csv` already assumed.
  This validates `b_extract.py`'s whole eager-attribution design end-to-end on real hardware, modulo the
  dlopen fix.

## 3. Real call-site instrumentation patch (not GPU-pilot-tested beyond import, see below)

`sglang_b_markers.patch` now has four hunks: the `frontier_b_markers.py` helper module (regenerated from
the current `marker_runtime.py`, which gained `begin_scope`/`end_scope`/`begin_forward`/`end_forward` —
module-level singleton + memoized labels; forward/scope nesting is resolved later by
`normalize_b.bind_labels` purely via timestamp containment, so call sites never thread a forward ID), and
three real call-site diffs against the exact recovered image source
(`evidence/cluster-inspection-21725-amd-mi355x-3/srt__*.py`, confirmed byte-identical to upstream
`v0.5.18` per the operator investigation's E5 for every file touched here):

| file | scopes added |
|---|---|
| `models/qwen3_moe.py` | `qkv_qk_norm` (compound, around `qkv_proj` GEMM + `apply_qk_norm`); `rope_kv` (fused, around the `rotary_emb` call that does RoPE+KV-write in one Triton kernel on HIP — the one physical unit mapping to both A's `attn_rope` and `attn_kv_cache_save`); `attention_prefill`/`attention_decode`/`attention_mixed` (around `self.attn(...)`, chosen per-call from `forward_batch.forward_mode`, preserving the image's extra output-fill kernel inside the boundary since it's not touched); `output_projection` (around `self.o_proj(...)` alone, `reduce_results=False` so no communication is captured); `B\|fwd\|<n>` forward boundary (around the single `self.model(...)` call in `Qwen3MoeForCausalLM.forward`) |
| `models/qwen2_moe.py` | `embedding` (around `self.embed_tokens(input_ids)`, once per forward, first-PP-rank-only path) |
| `layers/communicator.py` | `input_norm` (`norm_only` for the residual-is-None/layer-0 branch, `add_norm` for the TP>1-no-fusion and residual-present branches of `prepare_attn` — 3 real call sites, all direct inline calls); `post_norm` (`add_norm`, in both `_simple` (TP1) and `_gather_hidden_states_and_residual`'s non-DP AR+norm branch (TP4) of `CommunicateWithAllReduceAndLayerNormFn`, which `prepare_mlp` dispatches to; `layer_id` had to be stashed onto `self._context` in `prepare_mlp` since these are `@staticmethod`s with no `self`) |

Explicitly **not** instrumented (documented in code comments, not silently dropped): the AR-fusion and
DP-attention branches of `prepare_attn`/`prepare_mlp` (disabled by this recipe: "Leave all-reduce fusion
... disabled"), the CUDA-only `fused_qk_norm_rope` path (never executes on HIP), `aiter_backend.py`
internals (not needed — wrapping `self.attn(...)` at the call site already brackets every kernel it
dispatches, fill kernel included).

### Validation performed

- `patch -p1 --dry-run` and a full apply against fresh copies of the three recovered originals: clean,
  no fuzz, no offsets.
- `ast.parse` on all 4 resulting files: syntax OK.
- **Real GPU job** (`smatar-b-patchimport2`, node `amd-mi355x-3`, 1 GPU, approved image, patched files
  bind-mounted individually over their exact in-image paths): `import sglang.srt.models.qwen3_moe`,
  `.qwen2_moe`, `.layers.communicator` all succeed with `FRONTIER_B_MARKERS=1`/`FRONTIER_B_TRACER=roctx`
  set; `frontier_b_markers._BUFFER` is a real `MarkerBuffer` wrapping the fixed ROCTx library;
  `begin_scope`/`end_scope`/`begin_forward`/`end_forward` all callable without error. This caught and let
  me fix a real bug in my own process: the patch's first revision still carried a **stale**
  `frontier_b_markers.py` hunk (regenerated from `marker_runtime.py` before I'd added the
  scope/forward helpers), which the first import job correctly failed with
  `AttributeError: module ... has no attribute '_BUFFER'`. Rebuilt the combined patch from the current
  `marker_runtime.py` and the three already-correct call-site diffs; the re-run (`smatar-b-patchimport2`)
  passed clean. `evidence/patch_source_hashes.json` has the final original/patched/patch SHA-256s.
- `run_b.sbatch` changed from mounting a whole prepared package tree to mounting the 4 patched/new files
  individually at their exact container paths (`B_PATCHED_FRONTIER_B_MARKERS`, `B_PATCHED_QWEN3_MOE`,
  `B_PATCHED_QWEN2_MOE`, `B_PATCHED_COMMUNICATOR` env vars) — avoids staging a full ~3000-file sglang
  copy to change 4 files; this is what the import-validation job above exercises directly.
- Local test suite: added `test_rocprof_marker_csv_schema` (real schema, from the probe evidence) and
  `test_marker_runtime_scope_forward_helpers` (skips without `torch`, which isn't in the local dev venv)
  to `tests/test_collector.py`. **136 passed, 2 skipped** (pyarrow, torch — both expected locally, both
  present in the GPU image).

### What this does NOT establish (explicitly, so it isn't mistaken for a pilot)

- No SGLang server was started. No request was served. No scope actually fired against a real forward
  pass — the import test proves the patched module graph loads and the marker API is callable, not that
  the markers land in the right place relative to real kernels during real inference.
- CPU-side scheduler/request/forward metadata (b_check.py's independent "scheduler" records, sequence
  vectors, `global_forward_id` cross-rank correspondence) has **no hooks yet** — this instrumentation
  phase only covers the GPU-marker half of `03_COLLECTION_AGENT_PROMPT.md` §2. Needed next: hooks into
  SGLang's scheduler (`scheduler.py`/`schedule_batch.py`) to emit the independent records `b_check.py`
  requires; without them, `check_b.py` cannot reconcile anything from a real run, even once a real trace
  exists.
- Graph replay attribution is still unresolved (unchanged from before this phase).
- The `rope_kv`/`fused` variant assumes `enable_fused_set_kv_buffer(forward_batch) and
  self.compatible_with_fused_kv_buffer` is always true under this recipe (true for Qwen3/non-MRoPE on
  HIP per the operator investigation, no server flag disables it) — not independently re-verified by a
  real trace in this phase.

## Next concrete steps (in order)

1. Build the minimal SGLang scheduler hooks needed to emit `b_check.py`'s independent scheduler/request
   records (the actual remaining engineering gap for a checkable real bundle).
2. Resolve `server_argv`/`telemetry_argv`/effective flags for an actual TP1 eager launch (page size 1,
   radix cache off, `SGLANG_USE_AITER=1`, `--attention-backend aiter`, chunked-prefill 16384, the model
   path verified in §1) and fill `launch_plan.json` for real (currently still the placeholder).
3. Run the smallest possible real pilot (`run_b.sbatch`, TP1, 1 GPU, one short prompt) and push the
   resulting real bundle through `normalize_b.py` → `extract_b.py` → `check_b.py`.
4. Only after that pilot's results are presented: TP4, graph decode, chunking, mixed, Kineto cross-check,
   controls (pilot matrix A-G) — then the D1/D2/D3 sweep, per the task's own gating.
