# Analysis and probe scripts behind documents 09–12 (saved from the 2026-09-16…24 session scratchpad)

Run from the worktree root with `python3 <script> ...`; each header docstring says what it does. None is a checker; the gates live in
`../sanity_check.py` and `../test_measurement_validity.py`.

| script | used for |
|---|---|
| `curve_check.py <new.csv> [old.csv]` | per op × TP, lists unrecovered >15 % drops between consecutive token counts (the dip symptom) and the new/old median ratio by TP — the tables in `12_` §5 |
| `probe_summary.py <linear_op.csv>` | clock-probe medians/ranges, host enqueue vs wall, whole-forward closure per token count — `09_` §3a/§3b, run RUN.md tables |
| `rope_kernel_count.py <kineto_trace.json>` | kernels per `vidur_<scope>` annotation with the tracer's correlation matching — the "1 kernel per attn_rope" acceptance check (`11_`) |
| `rope_bench.py frontier_customop\|vllm_native\|vllm_aiter` (in the container, 1 GPU) | timing + numerics of the three RoPE paths — the benchmark table in the TASK-1 record |
| `sgl_rope_probe.py` (in the container) | SGLang's and AITER's rope objects on the same shapes — why the fused custom op is the reference |
| `rope_fallback_check.py` | CPU check that exposed the pre-fix fallback rotating only head 0 (now superseded by `tests/unit/test_rotary_embedding_numerics.py`) |
| `build_codex_plan_prompt.py <out.txt>` | assembles the Codex plan-review prompt from a plan file (ForgeLoop Stage 6b) |
