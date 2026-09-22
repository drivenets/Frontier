"""RecordFunctionTracer trace-file lifecycle (plan linear-op-recollection-contract-a, D5 / TASK-1).

A dense record_function run writes one ~4.9 MB chrome trace per shape (157 MB for 32 shapes measured 2026-09-17), i.e.
~65 GB for the 13,308-shape grid, so LinearOpWrapper asks the tracer to delete each trace after a successful parse.
The class default keeps traces (the attention and MoE profilers construct the tracer too and are out of scope).
"""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer  # noqa: E402
from tests.unit.test_linear_op_two_pass_orchestration import device  # noqa: E402,F401  (fake CUDA device fixture)


def _synthetic_trace(kernel_dur_us: float) -> dict:
    """One vidur_attn_rope annotation with one launch child correlated to one kernel event."""
    return {"traceEvents": [
        {"cat": "user_annotation", "name": "vidur_attn_rope", "ts": 100.0, "dur": 50.0},
        {"cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": 110.0, "dur": 5.0, "args": {"correlation": 7}},
        {"cat": "kernel", "name": "rotary_embedding_kernel", "ts": 200.0, "dur": kernel_dur_us, "args": {"correlation": 7}},
    ]}


def _tracer_with_trace(tmp_path: Path, kernel_dur_us: float = 12.0, **kwargs) -> RecordFunctionTracer:
    tracer = RecordFunctionTracer(str(tmp_path), **kwargs)
    Path(tracer.trace_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tracer.trace_path).write_text(json.dumps(_synthetic_trace(kernel_dur_us)))
    return tracer


def test_trace_kept_by_default(tmp_path):
    tracer = _tracer_with_trace(tmp_path)
    stats = tracer.get_operation_time_stats()
    assert stats["attn_rope"]["median"] == pytest.approx(0.012)
    assert os.path.exists(tracer.trace_path)


def test_trace_removed_after_successful_parse_when_not_kept(tmp_path):
    tracer = _tracer_with_trace(tmp_path, keep_trace=False)
    stats = tracer.get_operation_time_stats()
    assert stats["attn_rope"]["count"] == 1
    assert not os.path.exists(tracer.trace_path)


def test_trace_kept_when_parse_raises(tmp_path):
    tracer = _tracer_with_trace(tmp_path, kernel_dur_us=0.0, keep_trace=False, fail_on_zero_cuda_time=True)
    with pytest.raises(ValueError):
        tracer.get_operation_time_stats()
    assert os.path.exists(tracer.trace_path)  # the error message cites the trace path as evidence


def test_linear_op_wrapper_passes_keep_trace_from_env(device, monkeypatch, tmp_path):
    from frontier.profiling.linear_op import linear_op_wrapper as low
    from tests.unit.test_linear_op_two_pass_orchestration import _FakeTracer, _build_wrapper

    created = []

    class _Recording(_FakeTracer):
        def __init__(self, output_dir, **kwargs):
            super().__init__(output_dir)
            created.append(kwargs)

    monkeypatch.setattr(low, "RecordFunctionTracer", _Recording)
    plan = {"enabled_ops": ["attn_post_proj"], "padded_n_embd": 136, "padded_n_expanded_embd": 264}
    monkeypatch.delenv("FRONTIER_RF_KEEP_TRACES", raising=False)
    _build_wrapper(device, "record_function", tmp_path, profiling_plan=plan).profile(32)
    monkeypatch.setenv("FRONTIER_RF_KEEP_TRACES", "1")
    _build_wrapper(device, "record_function", tmp_path, profiling_plan=plan).profile(32)
    assert [c.get("keep_trace") for c in created] == [False, True]
