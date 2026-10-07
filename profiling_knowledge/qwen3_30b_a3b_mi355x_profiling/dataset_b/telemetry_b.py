"""External sysfs sampler and calibrated joins; never inserts GPU work."""
import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path
import re
import time

from b_common import indexed, integer, read_json, require


def parse_dpm_clock(raw):
    current = [line for line in raw.splitlines() if "*" in line]
    require(len(current) == 1, "no unique active DPM state")
    match = re.search(r"([0-9.]+)\s*Mhz", current[0], re.I)
    require(match is not None, "unknown DPM clock format")
    return float(match.group(1))


def sample_device(device, sample_id):
    """Host monotonic read interval brackets the sensor read, not its update."""
    start = time.monotonic_ns()
    raw, values, errors = {}, {}, []
    for key in ("sclk", "mclk"):
        try:
            raw[key] = Path(device[key + "_path"]).read_text()
            values[key + "_mhz"] = parse_dpm_clock(raw[key])
        except (OSError, ValueError) as e:
            errors.append(f"{key}: {e}")
            values[key + "_mhz"] = None
    end = time.monotonic_ns()
    return dict(sample_id=sample_id, gpu_id=device["gpu_id"],
                timebase="host_monotonic_ns", read_begin_ns=start, read_end_ns=end,
                read_latency_ns=end - start, sensor_resolution_ns=device.get("sensor_resolution_ns"),
                raw=raw, errors=errors, **values)


class ClockJoin:
    def __init__(self, samples, calibrations, policy):
        indexed(samples, ["sample_id"])
        self.calibrations = indexed(calibrations, ["gpu_id"])
        self.policy = policy
        self.by_gpu = {}
        for s in samples:
            integer(s["read_begin_ns"], "sample read begin")
            integer(s["read_end_ns"], "sample read end", s["read_begin_ns"])
            require(s["timebase"] == "host_monotonic_ns", "unknown sampler timebase")
            self.by_gpu.setdefault(s["gpu_id"], []).append(s)
        for gpu, rows in self.by_gpu.items():
            rows.sort(key=lambda s: s["read_begin_ns"] + s["read_end_ns"])
            self.by_gpu[gpu] = (rows, [(s["read_begin_ns"] + s["read_end_ns"]) // 2 for s in rows])

    def join(self, kernel):
        gpu = kernel["gpu_id"]
        result = dict(clock_status="missing_device", clock_sample_id=None)
        if gpu not in self.by_gpu or (gpu,) not in self.calibrations:
            return result
        c = self.calibrations[gpu,]
        if not self.policy.get("frozen") or not c.get("evidence_ref"):
            return dict(result, clock_status="uncalibrated")
        # Anchor-relative arithmetic avoids rounding large integer timestamp origins.
        scale = c["scale"]
        require(math.isfinite(scale) and scale > 0, "invalid clock scale")
        integer(c["uncertainty_ns"], "calibration uncertainty")
        host = c["host_anchor_ns"] + round((kernel["begin_ns"] - c["trace_anchor_ns"]) * scale)
        if not c["valid_trace_begin_ns"] <= kernel["begin_ns"] <= kernel["end_ns"] <= c["valid_trace_end_ns"]:
            return dict(result, clock_status="outside_calibration")
        rows, times = self.by_gpu[gpu]
        pos = bisect_left(times, host)
        # Both sides are needed to detect sampler gaps and window truncation.
        if pos == 0 or pos == len(rows):
            return dict(result, clock_status="unbracketed")
        before, after = rows[pos - 1], rows[pos]
        nearest = min((before, after), key=lambda s: abs(host - (s["read_begin_ns"] + s["read_end_ns"]) // 2))
        mid = (nearest["read_begin_ns"] + nearest["read_end_ns"]) // 2
        age = abs(host - mid)
        resolution = nearest["sensor_resolution_ns"]
        uncertainty = c["uncertainty_ns"] + (nearest["read_end_ns"] - nearest["read_begin_ns"] + 1) // 2
        result.update(clock_sample_id=nearest["sample_id"], clock_age_ns=age,
                      clock_alignment_uncertainty_ns=uncertainty,
                      clock_bracket_gap_ns=times[pos] - times[pos - 1],
                      clock_sensor_resolution_ns=resolution,
                      sclk_mhz=nearest["sclk_mhz"], mclk_mhz=nearest["mclk_mhz"],
                      clock_before_sample_id=before["sample_id"], clock_after_sample_id=after["sample_id"])
        status = "valid_context"
        if nearest.get("errors") or any(nearest[x] is None or not math.isfinite(nearest[x]) or nearest[x] <= 0
                                        for x in ("sclk_mhz", "mclk_mhz")):
            status = "sensor_error"
        elif resolution is None:
            status = "unknown_resolution"
        elif type(resolution) is not int or resolution < 0:
            status = "invalid_resolution"
        elif age + uncertainty + resolution > self.policy["max_age_ns"]:
            status = "stale"
        elif uncertainty > self.policy["max_uncertainty_ns"]:
            status = "uncertain_alignment"
        elif times[pos] - times[pos - 1] > self.policy["max_gap_ns"]:
            status = "sampler_gap"
        result["clock_status"] = status
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("devices", help="JSON list: gpu_id, sclk_path, mclk_path, sensor_resolution_ns")
    p.add_argument("output")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--interval-ms", type=float, default=100)
    args = p.parse_args()
    require(args.seconds > 0 and args.interval_ms > 0, "positive sampler duration/cadence required")
    devices = read_json(args.devices)
    require(devices, "empty device map")
    indexed(devices, ["gpu_id"])
    end = time.monotonic() + args.seconds
    n = 0
    with open(args.output, "x") as f:
        while time.monotonic() < end:
            start = time.monotonic()
            for d in devices:
                f.write(json.dumps(sample_device(d, f"sample:{n}"), allow_nan=False) + "\n")
                n += 1
            f.flush()  # sidecar process only; never the serving worker
            time.sleep(max(0, args.interval_ms / 1000 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
