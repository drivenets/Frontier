"""Independent scheduler/request reconciliation plus extraction acceptance checks."""
from collections import Counter, defaultdict
from b_common import VERSION, features, indexed, integer, interval, require
from b_extract import extract


def expected_scopes(forward, manifest, spec):
    expected = set()
    for rule in spec["rules"]:
        if forward["phase"] not in rule["phases"]:
            continue
        layers = {"model": [None], "all": range(manifest["model"]["num_layers"]),
                  "first": [0], "later": range(1, manifest["model"]["num_layers"])}[rule["layers"]]
        for layer in layers:
            expected.add((layer, rule["physical_scope"], rule["variant"]))
    return expected


def verify_forward_scope_metadata(bundle, spec):
    """Every forward's *metadata-recorded* scope invocations (bundle["scopes"],
    written by marker_runtime.begin_scope independently of whether the native
    trace or extractor later manage to attribute a kernel to it) must exactly
    equal what scope_spec.json's rules say must occur for that forward's
    phase and the model's layer count -- computed from first principles via
    expected_scopes(), not from anything the native trace or extract()
    observed.

    This is the one shared structural validator both check() and
    check_dataset_a_scope.py use (code review finding 1: a scope with no
    attributed kernel has no entry in extract()'s result["scopes"] summaries,
    so a check built only from that output can never detect a scope whose
    marker or dispatch the native trace lost -- the expected/actual
    comparison here never depends on attribution succeeding).

    Returns (forwards, scope_groups), both indexed by (process_id,
    forward_id), for the caller to reuse.
    """
    m = bundle["manifest"]
    forwards = indexed(bundle["forwards"], ["process_id", "forward_id"])
    scope_groups = defaultdict(list)
    for s in bundle["scopes"]:
        require((s["process_id"], s["forward_id"]) in forwards, "orphan scope forward")
        scope_groups[s["process_id"], s["forward_id"]].append(s)
    for key, f in forwards.items():
        actual = [(s["layer"], s["physical_scope"], s["variant"]) for s in scope_groups[key]]
        require(len(actual) == len(set(actual)), "duplicate physical scope invocation")
        require(set(actual) == expected_scopes(f, m, spec),
                "scope coverage disagrees with observed forward applicability")
    return forwards, scope_groups


def check(bundle, spec):
    m = bundle["manifest"]
    require(m["schema_version"] == VERSION, "schema version mismatch")
    require(m["scope_spec_version"] == spec["version"], "scope specification mismatch")
    require(m["data_kind"] in ("synthetic_fixture", "live"), "unknown evidence kind")
    integer(m["tp"], "TP", 1)
    integer(m["model"]["num_layers"], "layers", 1)
    procs = indexed(m["processes"], ["process_id"])
    for p in procs.values():
        integer(p["rank"], "TP rank")
    require(len(procs) == m["tp"] and {p["rank"] for p in procs.values()} == set(range(m["tp"])),
            "missing or duplicate TP rank")
    require(len({p["gpu_id"] for p in procs.values()}) == m["tp"], "ranks share GPU identity")
    require(m["windows"], "missing capture windows")
    windows = indexed(m["windows"], ["process_id"])
    require(set(windows) == set(procs), "missing rank trace window")
    for w in windows.values():
        interval(w)
        require(w["complete"] and w["drained"] and w["metadata_complete"], "incomplete trace or metadata window")
    diagnostics = indexed(m["trace_diagnostics"], ["process_id"])
    require(set(diagnostics) == set(procs), "missing trace diagnostics for rank")
    limitations = []
    for d in diagnostics.values():
        require(d["dropped_records"] in (0, None), "tracer reports dropped records")
        require(d["raw_kernel_count"] > 0, "empty trace diagnostics")
        if d["dropped_records"] is None:
            limitations.append(f"{d['process_id']}: tracer has no drop counter; scheduler/marker/count checks cannot prove no interior dispatch loss")
    require(bundle["kernels"] and bundle["scopes"] and bundle["forwards"], "empty collected data")
    counts = Counter(k["process_id"] for k in bundle["kernels"])
    require(set(counts) == {k[0] for k in procs}, "missing rank kernels")
    for key, d in diagnostics.items():
        require(counts[key[0]] == d["raw_kernel_count"], "raw kernel count mismatch")
    for k in bundle["kernels"]:
        w = windows[k["process_id"],]
        require(w["begin_ns"] <= k["begin_ns"] <= k["end_ns"] <= w["end_ns"], "kernel crosses capture window")
        require(k["raw_ref"], "missing kernel source reference")
    observed = indexed(bundle["scheduler"], ["global_forward_id"])
    require(observed, "missing independent scheduler observations")
    requests = indexed(bundle["requests"], ["request_id"])
    require(requests, "missing request completion ledger")
    scopes = indexed(bundle["scopes"], ["process_id", "scope_id"])
    forwards, scope_groups = verify_forward_scope_metadata(bundle, spec)
    cross_rank = defaultdict(list)
    derived_forwards = []
    for key, f in forwards.items():
        require((f["process_id"],) in procs, "unknown forward process")
        global_key = (f["global_forward_id"],)
        require(global_key in observed, "forward lacks scheduler evidence")
        obs = observed[global_key]
        require(f["sequences"] == obs["sequences"], "rank/forward membership differs from scheduler")
        require(f["phase"] == obs["phase"], "scheduler phase mismatch")
        require(f["execution"] in ("eager", "replay"), "unknown execution regime")
        derived_forwards.append(dict(f, run_id=m["run_id"], **features(
            f, m["model"], m["tp"], m.get("consumer_aliases", {}).get("kv_cache_prediction_granularity"))))
        cross_rank[global_key].append(f["process_id"])
    require(set(cross_rank) == set(observed), "scheduler forward missing from trace metadata")
    for pids in cross_rank.values():
        require(len(pids) == m["tp"] and set(pids) == {k[0] for k in procs}, "missing/duplicate cross-rank forward")
    history = defaultdict(list)
    indexed(bundle["scheduler"], ["ordinal"])
    for o in sorted(observed.values(), key=lambda r: r["ordinal"]):
        require(o["source_ref"], "scheduler lacks independent source reference")
        indexed(o["sequences"], ["request_id"])
        for seq in o["sequences"]:
            require((seq["request_id"],) in requests, "unknown observed request")
            history[seq["request_id"]].append(seq)
    require(set(history) == {r[0] for r in requests}, "missing request execution")
    for (rid,), request in requests.items():
        require(request["status"] == "complete", "failed/incomplete request")
        integer(request["prompt_tokens"], "prompt tokens", 1)
        integer(request["generated_tokens"], "generated tokens", 1)
        require(request["generated_tokens"] == request["requested_new_tokens"], "generation count mismatch")
        require(len(request["input_ids"]) == request["prompt_tokens"], "input token evidence mismatch")
        require(len(request["output_ids"]) == request["generated_tokens"], "missing output token evidence")
        prefix, prompt = 0, 0
        decodes = 0
        for seq in history[rid]:
            require(seq["prefix_len"] == prefix, "request prefix progression mismatch")
            if seq["kind"] == "prefill":
                require(decodes == 0, "prefill after decode")
                prompt += seq["query_len"]
            else:
                require(prompt == request["prompt_tokens"], "decode before prompt completion")
                decodes += 1
            prefix += seq["query_len"]
        require(prompt == request["prompt_tokens"], "request prompt not fully observed")
        # Not prefill-yields-token-1 + (generated-1) decodes: with overlap
        # scheduling (disable_overlap_schedule=False, this recipe's runtime
        # default) SGLang's one-step-ahead pipeline makes prefill produce no
        # directly-sampled token, so there is exactly one decode forward per
        # generated token. Confirmed on two independent requests with
        # different generated_tokens (jobs 21791 and 21796: warmup-0 with 2,
        # r-0 with 4 -- both decodes == generated_tokens exactly, not -1).
        require(decodes == request["generated_tokens"], "decode/first-token convention mismatch")
    # This convention must be verified on the live image; speculative decoding is not supported.
    result = extract(bundle)
    result["forwards"] = derived_forwards
    for s in result["scopes"]:
        rule = next(r for r in spec["rules"] if r["physical_scope"] == s["physical_scope"]
                    and r["variant"] == s["variant"])
        s.update(scope_spec_version=spec["version"], a_scopes=rule["a_scopes"],
                 mapping_status=rule["compatibility"])
    state_counts = Counter(r["attribution"] for r in result["kernels"])
    require(set(state_counts) <= {"excluded", "eager_correlation"}, "unresolved/ambiguous target attribution")
    allowed_exclusions = set(spec["exclusions"])
    for r in result["kernels"]:
        if r["attribution"] == "excluded":
            require(r["exclusion_reason"] in allowed_exclusions, "unknown exclusion reason")
        else:
            f = forwards[r["process_id"], r["forward_id"]]
            require(f["execution"] == "eager", "replay mislabeled as eager attribution")
    require({(s["process_id"], s["scope_id"]) for s in result["scopes"]} == set(scopes),
            "scope has no attributed dispatches")
    quality = Counter(r["clock_status"] for r in result["kernels"] if r["attribution"] != "excluded")
    require(set(quality) == {"valid_context"}, "invalid target clock coverage: " + str(dict(quality)))
    limitations.append("Clock values are sampled context, not instantaneous per-kernel clock measurements.")
    require(m["clock_policy"].get("evidence_ref"), "clock limits lack calibration evidence")
    if spec["status"] != "resolved":
        limitations.append("Physical/A mapping is provisional; no complete B coverage claim.")
    live_requirements = ("image_digest", "source_hashes", "patch_sha256", "effective_config",
                         "model_revision", "tokenizer_revision", "numerical_equivalence_ref",
                         "perturbation_ref", "tracer_crosscheck_ref", "rank_device_evidence_ref")
    missing = [key for key in live_requirements if not m.get(key)]
    # This version cannot certify full B: replay and cross-run TP/tracer coverage
    # are deliberately still release gates, even if a single eager run is sound.
    dataset_accepted = False
    limitations.append("b0.1 checks eager structure only; full Dataset B acceptance requires replay and cross-run pilot validation.")
    return result, dict(schema_version=VERSION, run_id=m["run_id"], structural_pass=True,
                        dataset_accepted=dataset_accepted, data_kind=m["data_kind"],
                        raw_kernels=len(result["kernels"]), target_kernels=state_counts["eager_correlation"],
                        excluded_kernels=state_counts["excluded"], scopes=len(result["scopes"]),
                        observed_forwards=len(observed), rank_forwards=len(forwards),
                        ranks=m["tp"], requests=len(requests), clock_states=dict(quality),
                        missing_live_evidence=missing, limitations=limitations)
