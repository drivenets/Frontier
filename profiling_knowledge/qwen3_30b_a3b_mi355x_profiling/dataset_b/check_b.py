"""Recompute checks from raw normalized inputs; never trusts extracted summaries."""
import argparse
import json
from pathlib import Path
from b_common import Invalid, read_json, write_json
from b_check import check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    p.add_argument("--scope-spec", default=str(Path(__file__).with_name("scope_spec.json")))
    p.add_argument("--report")
    p.add_argument("--require-dataset-accepted", action="store_true")
    args = p.parse_args()
    try:
        _, report = check(read_json(args.bundle), read_json(args.scope_spec))
    except (Invalid, KeyError, TypeError, ValueError, OSError) as e:
        report = dict(structural_pass=False, dataset_accepted=False, error=str(e))
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["structural_pass"] or (args.require_dataset_accepted and not report["dataset_accepted"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
