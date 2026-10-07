"""Canary check: are the ops Dataset A already has data for fully and
unambiguously attributed in this bundle?

This is deliberately narrower than b_check.py's full acceptance gate, which
requires every kernel in the bundle to be resolved or excluded. Right now we
only care about the ops scope_spec.json's (provisional) rules already map to
a Dataset A column (its "a_scopes" field) -- dense linear/attention ops.
Kernels outside that set (MoE expert routing/sorting/GEMM, sampling,
quantization, scheduler-support kernels) are deliberately ignored here, not
fixed or excluded. b_check.py is untouched; a bundle can pass this canary and
still fail (or not yet have run) the full gate.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from b_common import Invalid, read_json, require, write_json
from b_extract import extract


def check_dataset_a_scope(bundle, spec):
    require(spec["status"] == "provisional", f"unexpected scope_spec status: {spec['status']!r}")
    rule_keys = {(r["physical_scope"], r["variant"]) for r in spec["rules"]}
    dataset_a_scope_names = sorted({physical for physical, _variant in rule_keys})

    result = extract(bundle)
    scopes = {(s["process_id"], s["scope_id"]): s for s in result["scopes"]}

    # ambiguous/graph_unresolved kernels never carry scope identity (b_extract.py
    # only fills physical_scope/variant/scope_id in the eager_correlation and
    # excluded branches), so they could never match a Dataset-A rule below --
    # but both states are always a real correctness problem (overlapping
    # markers, or a replay this version cannot verify), never a legitimate
    # "no coverage yet" case. Fail on them here instead of letting them fall
    # silently into "out of scope".
    broken = Counter(k["attribution"] for k in result["kernels"] if k["attribution"] in ("ambiguous", "graph_unresolved"))
    require(not broken, f"structurally broken kernel attribution, cannot be out of scope: {dict(broken)}")

    def in_dataset_a(k):
        return k["scope_id"] is not None and (k["physical_scope"], k["variant"]) in rule_keys

    in_scope = [k for k in result["kernels"] if in_dataset_a(k)]
    out_of_scope = [k for k in result["kernels"] if not in_dataset_a(k)]
    require(in_scope, "no kernels found for any Dataset-A-mapped scope")

    state_counts = Counter(k["attribution"] for k in in_scope)
    require(set(state_counts) == {"eager_correlation"},
            f"Dataset-A-scope kernel attribution is not fully resolved: {dict(state_counts)}")

    covered = {(k["process_id"], k["scope_id"]) for k in in_scope}
    expected = {key for key, s in scopes.items() if (s["physical_scope"], s["variant"]) in rule_keys}
    missing = expected - covered
    require(not missing,
            f"{len(missing)} Dataset-A-mapped scope invocation(s) have zero attributed kernels: "
            f"{sorted(missing)[:5]}")

    return dict(
        structural_pass=True,
        dataset_a_scope_names=dataset_a_scope_names,
        in_scope_kernels=len(in_scope),
        in_scope_scope_invocations=len(expected),
        out_of_scope_kernels=len(out_of_scope),
        out_of_scope_attribution=dict(Counter(k["attribution"] for k in out_of_scope)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", help="normalized bundle JSON (normalize_b.py output)")
    p.add_argument("--scope-spec", default=str(Path(__file__).with_name("scope_spec.json")))
    p.add_argument("--output", help="optional: write the report JSON here")
    a = p.parse_args()
    try:
        report = check_dataset_a_scope(read_json(a.bundle), read_json(a.scope_spec))
    except (Invalid, KeyError, TypeError, ValueError) as e:
        p.exit(1, f"FAIL: {e}\n")
    print(json.dumps(report, indent=2))
    if a.output:
        write_json(a.output, report)


if __name__ == "__main__":
    main()
