"""Execute saved SGLang request cohorts and preserve every response/error.

HTTP concurrency is submission policy, never observed model batch size.
Transport injection makes driver behavior testable without network/GPU access.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from b_common import digest, indexed, integer, read_json, require, write_json


def validate(manifest):
    require(manifest["schema_version"] == "b-requests-0.1", "request schema mismatch")
    require(manifest["requests"], "empty request manifest")
    indexed(manifest["requests"], ["request_id"])
    integer(manifest["concurrency"], "concurrency", 1)
    require(manifest["timeout_s"] > 0, "invalid request timeout")
    require(manifest["vocab_size"] > 0, "missing vocabulary bound")
    for r in manifest["requests"]:
        require(r["input_ids"], "empty prompt")
        for token in r["input_ids"]:
            integer(token, "token ID")
            require(token < manifest["vocab_size"], "token ID outside vocabulary")
        integer(r["max_new_tokens"], "max_new_tokens", 1)
        integer(r["cohort"], "cohort")
        require(r["phase"] in ("warmup", "measurement"), "unknown request phase")
        require(r["offset_s"] >= 0, "negative submission offset")
    # Warmup is a separate drained cohort; no implicit profiler control requests.
    groups = {}
    for r in manifest["requests"]:
        groups.setdefault(r["cohort"], set()).add(r["phase"])
    require(all(len(phases) == 1 for phases in groups.values()), "mixed warmup/measurement cohort")
    phases = [next(iter(groups[k])) for k in sorted(groups)]
    require(phases == sorted(phases, key=lambda p: p == "measurement"), "warmup after measurement")


def http_transport(url, payload, timeout):
    req = Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def send(request, base_url, timeout, cohort_start, transport):
    time.sleep(max(0, cohort_start + request["offset_s"] - time.monotonic()))
    payload = dict(rid=request["request_id"], input_ids=request["input_ids"],
                   sampling_params=dict(max_new_tokens=request["max_new_tokens"],
                                        ignore_eos=True, temperature=0),
                   stream=False, return_logprob=True)
    row = dict(request_id=request["request_id"], phase=request["phase"], cohort=request["cohort"],
               input_ids=request["input_ids"], prompt_tokens=len(request["input_ids"]),
               requested_new_tokens=request["max_new_tokens"],
               send_monotonic_ns=time.monotonic_ns(), send_unix_ns=time.time_ns(), payload=payload)
    try:
        response = transport(base_url.rstrip("/") + "/generate", payload, timeout)
        row["response"] = response
        meta = response.get("meta_info", {})
        output = response.get("output_ids")
        if output is None and meta.get("output_token_logprobs") is not None:
            output = [value[1] for value in meta["output_token_logprobs"]]
        row.update(output_ids=output, generated_tokens=meta.get("completion_tokens"),
                   server_request_id=meta.get("id"))
        require(meta.get("id") in (None, request["request_id"]), "server request ID mismatch")
        require(isinstance(output, list), "response lacks output token evidence")
        require(len(output) == row["generated_tokens"] == request["max_new_tokens"],
                "incomplete generated token evidence")
        require(all(type(t) is int for t in output), "invalid output token IDs")
        row["status"] = "complete"
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
    row["done_monotonic_ns"] = time.monotonic_ns()
    row["done_unix_ns"] = time.time_ns()
    return row


def drive(manifest, base_url, transport=http_transport, phase="all"):
    validate(manifest)
    rows = []
    requests = [r for r in manifest["requests"] if phase == "all" or r["phase"] == phase]
    require(requests, "selected phase has no requests")
    for cohort in sorted({r["cohort"] for r in requests}):
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=manifest["concurrency"]) as pool:
            pending = [pool.submit(send, r, base_url, manifest["timeout_s"], start, transport)
                       for r in requests if r["cohort"] == cohort]
            rows.extend(f.result() for f in as_completed(pending))
        # Continue collecting all outcomes; a failed cell is preserved, never retried silently.
    return sorted(rows, key=lambda r: r["request_id"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest")
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--output", required=True)
    p.add_argument("--phase", choices=["all", "warmup", "measurement"], default="all")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    m = read_json(a.manifest)
    validate(m)
    if a.validate_only:
        print(f"valid: {len(m['requests'])} requests; concurrency={m['concurrency']} (not batch size)")
        return
    Path(a.output).mkdir(parents=True, exist_ok=False)
    write_json(Path(a.output) / "manifest.json", m)
    write_json(Path(a.output) / "input_hash.json", {"sha256": digest(a.manifest), "phase": a.phase})
    rows = drive(m, a.url, phase=a.phase)
    write_json(Path(a.output) / "requests.json", rows)
    if any(r["status"] != "complete" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
