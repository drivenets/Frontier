"""Concurrent shader-clock estimate and host-enqueue timestamp per timed pass (follow-up item 1 of the 2026-09-17 review:
"every row of every future dataset carries a concurrent clock estimate"; and the host-enqueue time tests the queue-backpressure
hypothesis for the F6 8-token residual).

Contract (LinearOpWrapper._timed_pass / profile):
- `_sclk_probe_mhz()` times a fixed PROBE_CYCLES spin with an event pair on the idle device and returns cycles / us = MHz.
- Each pass records the probe immediately before its timed loop (after warm-up + synchronize + mark_warmup_end, device idle)
  and immediately after its trailing synchronize: `sclk_mhz_legacy_start/end`, `sclk_mhz_backlog_start/end` (NaN when the pass did not run).
- `host_enqueue_per_forward_ms{,_backlog}` = host wall from the first timed forward's start to the end of the last forward's
  enqueue (before the trailing synchronize) / ACTIVE_STEPS; the existing host_wall fields keep including the synchronize.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from frontier.profiling.linear_op import linear_op_wrapper as low  # noqa: E402
from tests.unit.test_linear_op_two_pass_orchestration import _build_wrapper, device, rf_row  # noqa: E402,F401  (fake device + record_function fixtures)


def test_probe_reports_the_fake_device_clock(device):
    # the fake device runs at _DEVICE_CYCLES_PER_MS cycles per ms -> MHz = cycles_per_ms / 1000
    from tests.unit.test_linear_op_two_pass_orchestration import _DEVICE_CYCLES_PER_MS
    mhz = low._sclk_probe_mhz()
    assert mhz == pytest.approx(_DEVICE_CYCLES_PER_MS / 1000.0)
    assert device.sleep_calls[-1] == low.PROBE_CYCLES


def test_profile_records_clock_and_enqueue_fields(device, tmp_path):
    from tests.unit.test_linear_op_two_pass_orchestration import _DEVICE_CYCLES_PER_MS
    row = _build_wrapper(device, "cuda_event", tmp_path).profile(64)
    for key in ("sclk_mhz_legacy_start", "sclk_mhz_legacy_end", "sclk_mhz_backlog_start", "sclk_mhz_backlog_end",
                "host_enqueue_per_forward_ms", "host_enqueue_per_forward_ms_backlog"):
        assert key in row, key
        assert not math.isnan(row[key]), key
    for key in ("sclk_mhz_legacy_start", "sclk_mhz_legacy_end", "sclk_mhz_backlog_start", "sclk_mhz_backlog_end"):
        assert row[key] == pytest.approx(_DEVICE_CYCLES_PER_MS / 1000.0)
    # enqueue time excludes the trailing synchronize, so it cannot exceed the wall time of the same pass
    assert row["host_enqueue_per_forward_ms"] <= row["host_wall_per_forward_ms"]
    assert row["host_enqueue_per_forward_ms_backlog"] <= row["host_wall_per_forward_ms_backlog"]


def test_record_function_path_has_nan_clock_fields(rf_row):
    row, _ = rf_row
    for key in ("sclk_mhz_legacy_start", "sclk_mhz_legacy_end", "sclk_mhz_backlog_start", "sclk_mhz_backlog_end",
                "host_enqueue_per_forward_ms", "host_enqueue_per_forward_ms_backlog"):
        assert math.isnan(row[key]), key
