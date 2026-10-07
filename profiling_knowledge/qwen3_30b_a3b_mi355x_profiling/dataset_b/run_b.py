"""Bounded pilot runner, inside the approved container on a Slurm allocation.

Requires verified call-site integration, explicit profiler control payloads and
effective arguments in a pinned launch plan. The checked-in plan is NOT ready.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

from b_common import digest, read_json, require, write_json
from drive_b import http_transport, validate
from inspect_image_b import IMAGE

# Client-side HTTP timeout for the /frontier_b_drain call; longer than the
# server's own internal per-rank wait (tokenizer_manager.py's
# frontier_b_drain() defaults to 60s) so the client doesn't give up first.
FRONTIER_B_DRAIN_HTTP_TIMEOUT_S = 90


def preflight(plan):
    require(plan["approved_image"] == IMAGE, "unapproved serving image")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", plan.get("image_id", "")), "approved image digest/ID unresolved")
    if os.environ.get("B_RUNNING_IMAGE_ID"):
        require(plan["image_id"] == os.environ["B_RUNNING_IMAGE_ID"], "running container differs from pinned image")
    require(plan.get("callsite_patch_verified") is True, "call-site patch is not verified against pinned source")
    require(plan.get("native_trace_adapter_verified") is True, "native trace adapter not validated on image")
    require(plan.get("effective_flags_verified") is True, "serving flags/backend not verified")
    require(plan.get("metadata_drain_verified") is True, "worker metadata/trace drain hook unverified")
    require(plan["tracer"] in ("rocprofv3", "kineto"), "unknown tracer")
    require(plan["tp"] in (1, 4), "implementation pilot is TP1 or TP4")
    require(plan["environment"]["DEBUG_CLR_MAX_BATCH_SIZE"] == "1000000", "main hygiene flag missing")
    require(plan["environment"]["FRONTIER_B_MARKERS"] == "1", "markers must be explicitly enabled")
    require(plan["environment"]["FRONTIER_B_TRACER"] == ("roctx" if plan["tracer"] == "rocprofv3" else "kineto"),
            "marker/tracer mismatch")
    allowed = {"DEBUG_CLR_MAX_BATCH_SIZE", "FRONTIER_B_MARKERS", "FRONTIER_B_TRACER", "FRONTIER_B_OUTPUT",
               "HSA_ENABLE_COREDUMP", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "SGLANG_USE_AITER",
               "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"}
    require(set(plan["environment"]) <= allowed, "environment contains unreviewed/unnecessary keys")
    require(plan["source_hashes"], "missing source hashes")
    for path, sha in plan["source_hashes"].items():
        require(digest(path) == sha, f"source hash mismatch: {path}")
    require(Path(plan["model_path"]).is_dir(), "model path inaccessible")
    require(Path(plan["model_path"], "config.json").is_file(), "missing model config")
    require(plan["server_argv"] and plan["telemetry_argv"], "missing server/sampler commands")
    require(0 < plan["ready_timeout_s"] <= 600, "unbounded readiness wait")
    require(0 < plan["run_timeout_s"] <= 1200, "unbounded pilot duration")
    require(plan["scope_spec_status"] == "exploratory", "runner is for exploratory pilot only, not sweep")
    validate(read_json(plan["requests"]))


def stop_owned(process):
    """Signal only a new process group created by this runner."""
    if process is None or process.poll() is not None:
        return False
    forced = False
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        os.killpg(process.pid, sig)
        try:
            process.wait(timeout=timeout)
            return forced
        except subprocess.TimeoutExpired:
            forced = True
    return True


def run(plan, output):
    preflight(plan)
    require(os.environ.get("SLURM_JOB_ID"), "GPU serving must run in Slurm")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch_plan.json", plan)
    # The scheduler process (not this orchestrator, and not the HTTP API
    # process) flushes Dataset B's CPU metadata via atexit on clean shutdown;
    # it needs this directory to exist and be writable before the server
    # starts. Owned by this same container/root-squash identity, so a plain
    # mkdir suffices (no separate chmod needed, unlike run_dir itself).
    b_metadata_dir = output / "b_metadata"
    b_metadata_dir.mkdir()
    server = sampler = driver = None
    status = dict(status="failed", dataset_accepted=False)
    # One id per run, shared between the server's environment and the drain
    # call below; confirmed_drain() independently checks it against its own
    # copy (set from this same env var), so a plumbing mistake here is a hard
    # failure, not a silent mismatch.
    capture_id = f"{os.environ.get('SLURM_JOB_ID', 'nojob')}:{output.name}"
    env = os.environ.copy()
    env.update(plan["environment"])
    env["FRONTIER_B_OUTPUT"] = str(b_metadata_dir)
    env["FRONTIER_B_CAPTURE_ID"] = capture_id
    request_manifest = read_json(plan["requests"])
    deadline = time.monotonic() + plan["run_timeout_s"]
    # The GPU Slurm/run_b.sbatch actually allocated is only known at job
    # start, not launch-plan-authoring time; run_b.sbatch resolves it (by PCI
    # bus id, since ROCR/CUDA_VISIBLE_DEVICES are always "0" in-cgroup
    # regardless of which physical GPU that is) and passes it via
    # B_GPU_DEVICES_JSON. telemetry_argv may reference the literal
    # placeholder "__GPU_DEVICES_JSON__" in place of a path.
    # Both placeholders may appear as a substring within a larger path argument
    # (e.g. "__OUTPUT_DIR__/rocprof_trace" for rocprofv3's -d flag), not only
    # as a standalone token, since the run's actual output directory and the
    # allocated GPU's telemetry device JSON are both only known at job start
    # (SLURM_JOB_ID-dependent), not at plan-authoring time.
    def substitute(a):
        # __TASK_DIR__: the container's default CWD is the image's own /app,
        # not this collector's directory, so a bare relative script name in a
        # launch plan (e.g. "telemetry_b.py") silently fails with "No such
        # file or directory" -- __TASK_DIR__ resolves to run_b.py's own
        # directory, which is always correct since this file and
        # telemetry_b.py/drive_b.py are always bind-mounted together.
        a = a.replace("__TASK_DIR__", str(Path(__file__).parent))
        a = a.replace("__OUTPUT_DIR__", str(output))
        devices_json = os.environ.get("B_GPU_DEVICES_JSON")
        if devices_json:
            a = a.replace("__GPU_DEVICES_JSON__", devices_json)
        return a
    telemetry_argv = [substitute(a) for a in plan["telemetry_argv"]]
    server_argv = [substitute(a) for a in plan["server_argv"]]
    try:
        with (output / "server.log").open("x") as log, (output / "telemetry.log").open("x") as tlog:
            server = subprocess.Popen(server_argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            sampler = subprocess.Popen(telemetry_argv, stdout=tlog, stderr=subprocess.STDOUT, start_new_session=True)
            ready_deadline = min(deadline, time.monotonic() + plan["ready_timeout_s"])
            while True:
                require(server.poll() is None, "server exited during initialization")
                require(time.monotonic() < ready_deadline, "readiness timeout")
                try:
                    with urlopen(plan["base_url"].rstrip("/") + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            # Driver is a child process so the outer run timeout bounds stuck HTTP workers.
            for phase in ("warmup", "measurement"):
                if not any(r["phase"] == phase for r in request_manifest["requests"]):
                    continue
                if phase == "measurement":
                    for control in plan["before_measurement"]:
                        http_transport(plan["base_url"].rstrip("/") + control["path"], control["payload"], 10)
                driver = subprocess.Popen([sys.executable, str(Path(__file__).with_name("drive_b.py")),
                                           plan["requests"], "--url", plan["base_url"], "--phase", phase,
                                           "--output", str(output / phase)], start_new_session=True)
                code = driver.wait(timeout=max(1, deadline - time.monotonic()))
                require(code == 0, "request driver failed; preserve request records")
            for control in plan["after_measurement"]:
                http_transport(plan["base_url"].rstrip("/") + control["path"], control["payload"], 30)
            require(server.poll() is None, "server exited before requested shutdown")
            # Explicit, acknowledged end-of-measurement drain: every TP worker
            # closes its own capture window and snapshots its buffered Dataset B
            # metadata at a safe boundary (between forward passes, verified by
            # reading Scheduler.event_loop_normal/_overlap -- see
            # marker_runtime.py's module docstring), atomically, before this
            # function sends any termination signal. A failure here must not
            # be papered over: no signal is sent, and the real error is
            # preserved for diagnosis rather than retried silently.
            drain_response = http_transport(
                plan["base_url"].rstrip("/") + "/frontier_b_drain",
                {"capture_id": capture_id}, FRONTIER_B_DRAIN_HTTP_TIMEOUT_S,
            )
            write_json(output / "drain_acks.json", drain_response)
            require(drain_response.get("success") is True, "drain reported failure; see drain_acks.json")
            acks = drain_response.get("acks", [])
            require(len(acks) == plan["tp"], f"expected {plan['tp']} drain acks, got {len(acks)}")
            for ack in acks:
                require(ack["success"] is True, f"rank {ack.get('rank')} drain failed: {ack.get('error')}")
                require(ack["capture_id"] == capture_id, f"rank {ack.get('rank')} acked a different capture_id")
                require(ack["record_count"] > 0, f"rank {ack.get('rank')} drained zero records")
                require(not ack["overflow"], f"rank {ack.get('rank')} reported marker-buffer overflow")
            status.update(status="collected_unvalidated", capture_id=capture_id)
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
    finally:
        drained_ok = status["status"] == "collected_unvalidated"
        stop_owned(driver)
        status["forced_server_shutdown"] = stop_owned(server)
        stop_owned(sampler)
        # Two genuinely different conditions, previously conflated under one
        # "failed_trace_drain" status: metadata safety no longer depends on a
        # clean shutdown at all (confirmed_drain() already ran and persisted
        # it, atomically, before any signal was sent above). A forced
        # shutdown *after* a successful drain only risks the *native*
        # tracer's own finalization (verified separately, later, by
        # assemble_bundle.py matching native_pid/capture_id) -- it is not a
        # metadata loss and must not be reported as one.
        if status["forced_server_shutdown"]:
            status["status"] = "collected_forced_shutdown" if drained_ok else "failed_before_drain"
        write_json(output / "status.json", status)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan")
    p.add_argument("--output")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    plan = read_json(a.plan)
    if a.preflight_only:
        preflight(plan)
        print("local preflight passed; effective runtime evidence still required")
    else:
        require(a.output, "new output directory required")
        status = run(plan, a.output)
        print(json.dumps(status))
        # "collected_forced_shutdown" still means confirmed_drain() succeeded
        # before any signal was sent -- metadata is safe; only the native
        # tracer's own finalization needs the separate, later check in
        # assemble_bundle.py. Anything else (including "failed_before_drain")
        # means the drain itself never completed.
        if status["status"] not in ("collected_unvalidated", "collected_forced_shutdown"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
