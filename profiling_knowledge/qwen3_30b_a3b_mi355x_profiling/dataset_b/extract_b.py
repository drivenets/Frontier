"""Extract immutable normalized B inputs; failed checks retain a failure report."""
import argparse
from pathlib import Path
from b_common import VERSION, Invalid, digest, read_json, write_json
from b_check import check


def run(source, output, spec_path, parquet=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "inputs.json", {"bundle": str(source), "bundle_sha256": digest(source),
                                      "scope_spec": str(spec_path), "scope_spec_sha256": digest(spec_path),
                                      "extractor_version": VERSION,
                                      "extractor_sources_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}})
    try:
        result, report = check(read_json(source), read_json(spec_path))
        if parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
        for name, rows in result.items():
            write_json(output / (name + ".json"), rows)
            if parquet:
                # Native payload stays in immutable inputs; nested variable native structs
                # are stored as JSON in Parquet while normalized columns remain typed.
                import json
                records = [dict(r, native=json.dumps(r.get("native"), sort_keys=True)) for r in rows]
                pq.write_table(pa.Table.from_pylist(records), output / (name + ".parquet"), compression="zstd")
        report["parquet_written"] = parquet
        write_json(output / "check.json", report)
        return report
    except (Invalid, KeyError, TypeError, ValueError, ImportError, OSError) as e:
        write_json(output / "failure.json", {"structural_pass": False, "error": str(e)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    p.add_argument("output", help="new directory; never overwrite an extraction")
    p.add_argument("--scope-spec", default=str(Path(__file__).with_name("scope_spec.json")))
    p.add_argument("--parquet", action="store_true", help="requires pyarrow; never silently falls back")
    a = p.parse_args()
    try:
        r = run(a.bundle, a.output, a.scope_spec, a.parquet)
        print(f"structural_pass={r['structural_pass']} dataset_accepted={r['dataset_accepted']} kernels={r['raw_kernels']}")
    except (Invalid, KeyError, TypeError, ValueError, ImportError, OSError) as e:
        p.exit(1, f"FAIL: {e}\n")


if __name__ == "__main__":
    main()
