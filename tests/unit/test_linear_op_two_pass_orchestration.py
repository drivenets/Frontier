"""CPU tests for the two-pass (legacy + GPU-bound) orchestration in LinearOpWrapper.

Companion to test_linear_op_timing_columns.py, which covers the pure helpers on CPU and the wrapper only on a GPU.
Here the CUDA primitives the wrapper and CudaTimer touch (torch.cuda.Event, torch.cuda.synchronize, torch.cuda._sleep)
are replaced by a fake device clock, so _enqueue_gpu_backlog, _timed_pass and the two-pass part of profile() run on a
CPU-only box. The wrapper is built via __new__ with just the attributes profile() reads; no GPTModel is constructed.

Fake device model: every Event.record() stamps the fake clock; the fake model advances the clock inside its CudaTimer
scopes (op durations), and the fake torch.cuda._sleep advances it by cycles / _DEVICE_CYCLES_PER_MS. The legacy pass
reports attn_post_proj at 0.036 ms (host launch span) and the GPU-bound pass at 0.010 ms (kernel), while attn_rope is
0.050 ms in both, so exactly one op is flagged legacy-host-bound.
"""
import gc
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.linear_op import linear_op_wrapper as low
from frontier.profiling.linear_op import spike_diag
from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
from frontier.profiling.utils.replicated_ops import split_replicated_result
from frontier.profiling.utils.singleton import Singleton

_DEVICE_CYCLES_PER_MS = 250_000.0  # the fake GPU's true clock: 20_000_000 calibration cycles -> 80 ms
_STALE_CYCLES_PER_MS = 200_000.0   # a stale module calibration -> requested spins under-deliver by 20 % (like job 21376)
LEGACY_MS = {"attn_post_proj": 0.036, "attn_rope": 0.050}
GPU_BOUND_MS = {"attn_post_proj": 0.010, "attn_rope": 0.050}


class _FakeDevice:
    """Fake device clock plus the call log the tests assert on."""

    def __init__(self):
        self.now_ms = 0.0
        self.sleep_calls = []        # cycles passed to torch.cuda._sleep, in order
        self.synchronize_calls = 0
        self.gc_enabled_in_forward = []

    @property
    def spin_calls(self):
        """torch.cuda._sleep calls that are calibration or backlog spins (the fixed-cycle clock probes are excluded)."""
        return [c for c in self.sleep_calls if c != low.PROBE_CYCLES]

    @property
    def backlog_enqueued(self):
        return bool(self.spin_calls)

    def make_event_class(self):
        device = self

        class FakeEvent:
            def __init__(self, enable_timing=False, **_):
                self.stamp = None

            def record(self, stream=None):
                self.stamp = device.now_ms

            def elapsed_time(self, end):
                assert self.stamp is not None and end.stamp is not None, "elapsed_time on an unrecorded event"
                return end.stamp - self.stamp

        return FakeEvent

    def sleep(self, cycles):
        self.sleep_calls.append(cycles)
        self.now_ms += cycles / _DEVICE_CYCLES_PER_MS

    def synchronize(self):
        self.synchronize_calls += 1

    def model(self, input_ids, positions):
        """Two timed ops per forward; attn_post_proj shrinks once the GPU runs behind the host (backlog enqueued)."""
        assert input_ids.device.type == "cpu" and positions.device.type == "cpu"
        self.gc_enabled_in_forward.append(gc.isenabled())
        durations = GPU_BOUND_MS if self.backlog_enqueued else LEGACY_MS
        for op, ms in durations.items():
            with CudaTimer(op):
                self.now_ms += ms


class _TorchProxy:
    """The wrapper module's `torch`: identical to torch except that device="cuda" is dropped from tensor factories."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def randint(*args, **kwargs):
        kwargs.pop("device", None)
        return torch.randint(*args, **kwargs)

    @staticmethod
    def tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return torch.tensor(*args, **kwargs)


def _model_config():
    return SimpleNamespace(
        max_position_embeddings=64, num_q_heads=8, num_kv_heads=2, embedding_dim=128, mlp_hidden_dim=256,
        vocab_size=1024, use_gated_mlp=True, use_qk_norm=True, model_arch="fake", share_expert_dim=None, share_q_dim=None,
        get_model_architecture_profile=lambda: SimpleNamespace(profile_id="fake-profile"),
    )


def _build_wrapper(device, profile_method, tmp_path, profiling_plan=None):
    wrapper = LinearOpWrapper.__new__(LinearOpWrapper)
    wrapper.profile_method = profile_method
    wrapper.timer_stats_store = TimerStatsStore(profile_method=profile_method)
    wrapper.model = device.model
    wrapper.model_config = _model_config()
    wrapper.num_tensor_parallel_workers = 2
    wrapper.padded_vocab_size = 1024
    wrapper.rank = 0
    wrapper.output_dir = str(tmp_path)
    wrapper.profiling_plan = profiling_plan
    return wrapper


@pytest.fixture
def device(monkeypatch):
    dev = _FakeDevice()
    monkeypatch.setattr(Singleton, "_instances", {})  # fresh TimerStatsStore per test; restored afterwards
    monkeypatch.setattr(torch.cuda, "Event", dev.make_event_class())
    monkeypatch.setattr(torch.cuda, "synchronize", dev.synchronize)
    monkeypatch.setattr(torch.cuda, "_sleep", dev.sleep)
    monkeypatch.setattr(low, "torch", _TorchProxy())
    monkeypatch.setattr(low, "_GPU_BACKLOG_MS", 0.0)
    monkeypatch.setattr(low, "_SLEEP_CYCLES_PER_MS", _STALE_CYCLES_PER_MS)
    return dev


@pytest.fixture
def row(device, tmp_path):
    return _build_wrapper(device, "cuda_event", tmp_path).profile(64)


# ---------------------------------------------------------------- profile(): two passes, schema, flags

def test_two_passes_share_warmup_and_timed_counts(row):
    for op in LEGACY_MS:
        for col in (row["time_stats"], row["time_stats_hostbound"]):
            assert col[op]["count"] == 50 and col[op]["warmup_count"] == 3, op
    assert row["warmup_steps"] == 3 and row["active_steps"] == 50


def test_forward_span_only_in_gpu_bound_column(row):
    assert "forward_gpu_span" in row["time_stats"]
    assert "forward_gpu_span" not in row["time_stats_hostbound"]
    span = row["time_stats"]["forward_gpu_span"]
    assert span["count"] == 50 and span["warmup_count"] == 3
    assert span["median"] == pytest.approx(sum(GPU_BOUND_MS.values()))  # one pair around the whole forward
    assert "forward_gpu_span" not in row["legacy_host_bound_ratio"]


def test_columns_carry_legacy_and_gpu_bound_medians(row):
    for op in LEGACY_MS:
        assert row["time_stats_hostbound"][op]["median"] == pytest.approx(LEGACY_MS[op]), op
        assert row["time_stats"][op]["median"] == pytest.approx(GPU_BOUND_MS[op]), op


def test_legacy_host_bound_flag_set_for_one_op_only(row):
    assert row["legacy_host_bound_ratio"]["attn_post_proj"] == pytest.approx(3.6)
    assert row["legacy_host_bound_ratio"]["attn_rope"] == pytest.approx(1.0)
    assert row["legacy_host_bound"] == {"attn_post_proj": True, "attn_rope": False}


def test_backlog_enqueued_once_and_only_in_second_pass(row, device):
    assert len(device.spin_calls) == 1  # calibration skipped (module constant set), spin only before pass-2 timed loop
    # legacy pass: 3 warm-up + 50 timed forwards, all at legacy durations; gpu-bound pass: warm-ups still legacy (spin
    # not yet enqueued), the 50 timed forwards at kernel durations.
    samples_legacy = _samples(row["time_stats_hostbound"]["attn_post_proj"])
    samples_gpu = _samples(row["time_stats"]["attn_post_proj"])
    assert samples_legacy == [pytest.approx(0.036)] * 53
    assert samples_gpu[:3] == [pytest.approx(0.036)] * 3 and samples_gpu[3:] == [pytest.approx(0.010)] * 50


def _samples(stat):
    return json.loads(stat["samples"])


def test_requested_backlog_defaults_to_factor_times_legacy_loop_wall(row):
    assert low.BACKLOG_REQUEST_FACTOR == 4.0
    legacy_loop_wall_ms = row["host_wall_per_forward_ms"] * 50
    assert legacy_loop_wall_ms > 0
    assert row["gpu_backlog_ms"] == pytest.approx(4.0 * legacy_loop_wall_ms)
    assert row["host_wall_per_forward_ms_backlog"] > 0


def test_requested_backlog_honours_env_override(device, tmp_path, monkeypatch):
    monkeypatch.setattr(low, "_GPU_BACKLOG_MS", 123.0)  # what FRONTIER_GPU_BACKLOG_MS=123 sets at import
    out = _build_wrapper(device, "cuda_event", tmp_path).profile(64)
    assert out["gpu_backlog_ms"] == 123.0
    assert device.spin_calls == [int(123.0 * _STALE_CYCLES_PER_MS)]


def test_actual_backlog_is_event_measured_and_refits_calibration(row, device):
    (cycles,) = device.spin_calls
    assert cycles == int(row["gpu_backlog_ms"] * _STALE_CYCLES_PER_MS)
    expected_actual = cycles / _DEVICE_CYCLES_PER_MS  # the spin under-delivers by 20 % against the stale calibration
    assert row["gpu_backlog_ms_actual"] == pytest.approx(expected_actual)
    assert row["gpu_backlog_ms_actual"] == pytest.approx(0.8 * row["gpu_backlog_ms"], rel=1e-3)  # int() truncation of cycles
    assert low._SLEEP_CYCLES_PER_MS == pytest.approx(_DEVICE_CYCLES_PER_MS)  # re-fitted from the measured spin


def test_timer_store_cleared_after_profile(row):
    assert TimerStatsStore(profile_method="cuda_event").TIMING_STATS == {}


def test_gc_disabled_inside_passes_and_restored(row, device):
    assert device.gc_enabled_in_forward and not any(device.gc_enabled_in_forward)
    assert gc.isenabled()


def test_static_metadata_passthrough(row):
    assert row["num_tokens"] == 64 and row["num_tensor_parallel_workers"] == 2
    assert row["n_head"] == 8 and row["n_kv_head"] == 2 and row["n_embd"] == 128 and row["n_expanded_embd"] == 256
    assert row["padded_n_embd"] == 128 and row["padded_n_expanded_embd"] == 256  # no profiling plan -> model dims
    assert row["model_architecture_profile"] == "fake-profile" and row["use_qk_norm"] is True


# ---------------------------------------------------------------- _enqueue_gpu_backlog: calibration branch

def test_enqueue_gpu_backlog_calibrates_on_first_use(device, monkeypatch, capsys):
    monkeypatch.setattr(low, "_SLEEP_CYCLES_PER_MS", None)
    start, end, cycles = low._enqueue_gpu_backlog(40.0)
    assert device.spin_calls[:2] == [20_000_000, 20_000_000]  # two calibration spins: the first warms the clock, the second is fitted
    assert low._SLEEP_CYCLES_PER_MS == pytest.approx(_DEVICE_CYCLES_PER_MS)
    assert cycles == int(40.0 * _DEVICE_CYCLES_PER_MS) and device.spin_calls[2] == cycles
    assert start.elapsed_time(end) == pytest.approx(40.0)
    assert device.synchronize_calls == 3  # one before and one after each calibration spin; the caller owns the trailing sync
    assert "cycles_per_ms=250000" in capsys.readouterr().out


def test_enqueue_gpu_backlog_skips_calibration_when_fitted(device):
    start, end, cycles = low._enqueue_gpu_backlog(12.0)  # 12 ms: 10 ms x the stale rate would equal PROBE_CYCLES
    assert device.spin_calls == [cycles] and cycles == int(12.0 * _STALE_CYCLES_PER_MS)
    assert device.synchronize_calls == 0
    assert start.elapsed_time(end) == pytest.approx(cycles / _DEVICE_CYCLES_PER_MS)


def test_profile_calibrates_once_across_tasks(device, tmp_path, monkeypatch):
    monkeypatch.setattr(low, "_SLEEP_CYCLES_PER_MS", None)
    wrapper = _build_wrapper(device, "cuda_event", tmp_path)
    wrapper.profile(64)
    wrapper.profile(128)
    assert len(device.spin_calls) == 4 and device.spin_calls[:2] == [20_000_000, 20_000_000]  # 2 calibration spins once + 1 spin per task


# ---------------------------------------------------------------- _timed_pass directly

def test_timed_pass_without_backlog_reports_zero_actual(device, tmp_path):
    wrapper = _build_wrapper(device, "cuda_event", tmp_path)
    ids = torch.zeros(4, dtype=torch.long)
    out = wrapper._timed_pass(ids, ids, backlog_ms=0.0, record_forward_span=False)
    assert out["backlog_actual_ms"] == 0.0 and device.spin_calls == []
    assert set(out["time_stats"]) == set(LEGACY_MS) and out["loop_wall_ms"] > 0
    assert low._SLEEP_CYCLES_PER_MS == _STALE_CYCLES_PER_MS  # nothing measured -> no re-fit


def test_timed_pass_zero_length_spin_keeps_calibration(device, tmp_path, monkeypatch):
    monkeypatch.setattr(device, "sleep", lambda cycles: device.sleep_calls.append(cycles))  # spin measures 0 ms
    monkeypatch.setattr(torch.cuda, "_sleep", device.sleep)
    wrapper = _build_wrapper(device, "cuda_event", tmp_path)
    ids = torch.zeros(4, dtype=torch.long)
    out = wrapper._timed_pass(ids, ids, backlog_ms=5.0, record_forward_span=True)
    assert out["backlog_actual_ms"] == 0.0
    assert low._SLEEP_CYCLES_PER_MS == _STALE_CYCLES_PER_MS  # guard: no division by a 0 ms spin


def test_timed_pass_records_spike_diag_hooks_when_enabled(device, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(spike_diag, "enabled", True)
    monkeypatch.setattr(spike_diag, "forward_begin", lambda: calls.append("b"))
    monkeypatch.setattr(spike_diag, "forward_end", lambda: calls.append("e"))
    wrapper = _build_wrapper(device, "cuda_event", tmp_path)
    ids = torch.zeros(4, dtype=torch.long)
    wrapper._timed_pass(ids, ids, backlog_ms=0.0, record_forward_span=False)
    assert calls == ["b", "e"] * 53  # one begin/end pair around every warm-up and timed forward


# ---------------------------------------------------------------- record_function path: single pass, NaN/empty new fields

class _FakeTracer:
    created = []

    def __init__(self, output_dir):
        self.output_dir = output_dir
        _FakeTracer.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_operation_time_stats(self, debug=False):
        return {"attn_post_proj": {"median": 0.01, "count": 50, "warmup_count": 0}}


@pytest.fixture
def rf_row(device, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(low, "RecordFunctionTracer", _FakeTracer)
    monkeypatch.setattr(_FakeTracer, "created", [])
    plan = {"enabled_ops": ["attn_post_proj", "attn_rope"], "padded_n_embd": 136, "padded_n_expanded_embd": 264}
    out = _build_wrapper(device, "record_function", tmp_path, profiling_plan=plan).profile(32)
    return out, capsys.readouterr().out


def test_record_function_path_runs_one_untimed_pass_and_no_backlog(rf_row, device):
    out, _ = rf_row
    assert device.spin_calls == []  # no spin on this path
    assert len(_FakeTracer.created) == 1
    assert out["time_stats"] == {"attn_post_proj": {"median": 0.01, "count": 50, "warmup_count": 0}}


def test_record_function_path_returns_empty_two_pass_fields(rf_row):
    out, _ = rf_row
    assert out["time_stats_hostbound"] == {} and out["legacy_host_bound_ratio"] == {} and out["legacy_host_bound"] == {}
    for key in ("host_wall_per_forward_ms", "host_wall_per_forward_ms_backlog", "gpu_backlog_ms", "gpu_backlog_ms_actual"):
        assert math.isnan(out[key]), key


def test_record_function_path_warns_on_missing_plan_ops_and_uses_plan_padding(rf_row):
    out, printed = rf_row
    assert "Missing operations: ['attn_rope']" in printed
    assert out["padded_n_embd"] == 136 and out["padded_n_expanded_embd"] == 264


# ---------------------------------------------------------------- split_replicated_result on a record_function-style row

def test_split_replicated_result_skips_non_dict_per_op_fields():
    result = {"num_tensor_parallel_workers": 2, "time_stats": {"emb": {"median": 1}, "attn_post_proj": {"median": 2}},
              "legacy_host_bound_ratio": float("nan")}  # a flattened / NaN field must pass through untouched
    sharded, replicated = split_replicated_result(result, {"emb"})
    assert set(sharded["time_stats"]) == {"attn_post_proj"} and set(replicated["time_stats"]) == {"emb"}
    assert math.isnan(sharded["legacy_host_bound_ratio"]) and math.isnan(replicated["legacy_host_bound_ratio"])
    assert "time_stats_hostbound" not in sharded  # absent fields are not invented
