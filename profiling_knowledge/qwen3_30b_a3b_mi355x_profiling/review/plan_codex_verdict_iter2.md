OVERALL: FAIL
BLOCKING_FINDINGS:
  - BLOCKER: D-7/TASK-3 attention grid validation — `get_attention_input_combinations(..., enable_chunked_prefill_grid_search=True)` emits duplicate standard feature keys; the plan says 2,772 standard rows per TP but also requires no duplicate feature-key rows, and `ExpectedAttentionGrid.standard_keys` is a `frozenset` that collapses them — Add an explicit final-dataset dedupe/normalization step and revise point counts/tests to distinguish raw profiler attempts from unique final rows.
  - BLOCKER: TASK-3 tests/expectations — `test_qwen3_manifest_matches_expectation_selection` requires `selected_operators` to be a subset of manifest profiling names, but `attn_pre_proj`, `attn_rope`, and `attn_post_proj` are not returned by `build_operator_manifest`; they come from `ATTENTION_LINEAR_OPS` — Split manifest operators from auxiliary predictor-support operators, or validate against `manifest profiling names ∪ ATTENTION_LINEAR_OPS`.
  - BLOCKER: D-7 linear_op validation — `linear_op.main` splits replicated ops for TP>1, so valid `linear_op.csv` rows have NaN medians for `emb`, `input_layernorm`, and `post_attention_layernorm` at TP 2/4/8; the plan’s global “no NaN medians” check will fail real output — Scope checks by operator/effective TP: replicated memory ops non-null at TP=1 only, attention-linear ops non-null at TP {1,2,4,8}.
  - BLOCKER: D-7 value-set validation — true-mixed rows set `batch_size = total_batch_size`, so valid values like 3, 5, 9, 17, 65, and 97 are not in `--batch_size_list`; the plan’s `batch_size ⊆ list` check will reject conforming data — Apply `batch_size` list checks only to standard rows; validate true-mixed rows using `num_prefill_seqs`, `decode_batch_size`, `decode_kv_cache_sizes`, and `total_batch_size <= 128`.
NON_BLOCKING_FINDINGS:
  - LOW: TASK-4 dry-run test — “no `TORCH_SDPA`” is ambiguous because the reused gpt-oss script header/default text mentions TORCH_SDPA even when `BACKENDS=AITER`; assert no TORCH_SDPA profiling command instead.
CHECKLIST_RESULTS:
  - REQ coverage: FAIL — REQ-12 is not satisfied because the planned validator rejects real duplicate/linear split/true-mixed output shapes.
  - Source traceability: FAIL — Duplicate-free standard grids, manifest-subset `selected_operators`, global linear no-NaN checks, and true-mixed `batch_size` checks contradict the checked source behavior.
  - Unresolved decisions: FAIL — The plan still needs explicit decisions for standard attention duplicate canonicalization and per-operator validation scoping.
  - Placeholder scan: PASS — No unresolved TBD-style placeholders beyond acceptable implementation-loop pseudocode.
  - Handoff completeness: FAIL — Several planned tests and acceptance checks would fail against real profiler outputs.
FIX_BRIEF:
  - D-7/TASK-3: add standard attention duplicate normalization and update raw-vs-unique point counts.
  - TASK-3: split `selected_operators` into manifest and attention-linear support sets, or change the subset assertion source.
  - D-7: scope linear no-NaN checks by effective TP and true-mixed value checks by true-mixed fields.
