"""End-to-end entrypoint: real sglang/vLLM sweep -> Frontier sim sweep -> comparison report."""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import List

from tools.validation.compare_plots import RunIdentity, write_html_report
from tools.validation.frontier_cli_translator import SimPoint, Topology, build_sweep
from tools.validation.metrics_extractor import SimResult, extract_sim_result, find_run_dir
from tools.validation.real_log_aggregator import load_and_aggregate
from tools.validation.real_log_parser import engine_label


def run_sim_point(sim_point: SimPoint, *, output_dir: str, model_name: str, log_dir: Path) -> SimResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sim_point.run_id}.log"
    with open(log_path, "w") as f:
        result = subprocess.run(sim_point.command, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"frontier.main failed for {sim_point.run_id} (see {log_path})")

    run_dir = find_run_dir(output_root=output_dir, model_name=model_name, run_id=sim_point.run_id)
    return extract_sim_result(
        run_dir,
        run_id=sim_point.run_id,
        concurrency=sim_point.concurrency,
        num_prompts=sim_point.real.num_prompts,
        calibrated_qps=sim_point.calibrated_qps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        help="Real benchmark_results/run_<label>/ directory, or a directory containing several "
        "repeated run_<label>/ subdirectories (same benchmark config, launched multiple times) "
        "-- when given a group, real-side metrics are aggregated (mean ± std) across every "
        "repetition found. A single run is treated as a group of one (no error bars).",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--attn-tp", type=int, required=True)
    parser.add_argument("--attn-dp", type=int, default=1)
    parser.add_argument("--moe-tp", type=int, default=0, help="defaults to --attn-tp if unset")
    parser.add_argument("--moe-ep", type=int, default=1)
    parser.add_argument("--pipeline-stages", type=int, default=1)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument(
        "--cc-backend", default="collective_sim",
        choices=["collective_sim", "astra_sim_analytical", "analytical", "vidur", "aiconfigurator"],
        help="Collective-communication cost model. Default collective_sim: the real compiled "
             "htsim-based topology simulator (frontier/cc_backend/backends/collective-sim/, a "
             "git submodule -- must be built first, see its README) -- meaningfully more "
             "accurate than the alternatives below for any topology with attn-tp/attn-dp/"
             "moe-tp/moe-ep > 1 (irrelevant for single-GPU runs, where every backend here "
             "short-circuits collective cost to 0 the same way). 'analytical' is a closed-form "
             "bandwidth+latency formula that ignores real profiled network data entirely -- "
             "confirmed to badly misestimate TP=8 collective cost on mi355x (a ~37ms constant "
             "regardless of actual data size); keep it only as a cheap fallback when "
             "collective-sim isn't built. 'astra_sim_analytical' is a similarly-named but "
             "unrelated pure-Python formula backend (no compiled dependency, easy to confuse "
             "with collective_sim/htsim -- see profiling_knowledge for the distinction). "
             "'vidur' trains sklearn models on data/profiling/network/{NETWORK_DEVICE}/"
             "{all_reduce,send_recv}.csv -- use it if you specifically want predictions from "
             "your own profiled hardware data rather than a topology simulator.",
    )
    parser.add_argument("--network-device", default=None, help="e.g. mi355x_8gpu (required for --cc-backend vidur)")
    parser.add_argument("--request-mode", default="closed_loop", choices=["closed_loop", "poisson"],
                         help="closed_loop (default, matches real max-concurrency semantics) or poisson (legacy calibrated-QPS approximation)")
    parser.add_argument("--block-size", type=int, default=32,
                         help="Must match the paged-attention block size the model's profiling data was collected "
                              "at (check data/profiling/compute/<device>/<model>/attention.csv's block_size column) "
                              "-- otherwise the execution-time predictor's training data comes back empty. "
                              "Default 32 matches deepseek-r1-0528; e.g. qwen3-a3b-30b-moe/gpt-oss-* use 16.")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True,
                         help="Default on, matching real vLLM/sglang server behavior. Disable only as a workaround "
                              "when the model's profiling data has zero is_mixed_batch rows (check "
                              "data/profiling/compute/<device>/<model>/attention.csv) -- chunked prefill lets "
                              "prefill and decode share a batch, which needs a trained attn_decode_in_mixed "
                              "predictor; without it, true mixed batches raise instead of silently mispredicting.")
    parser.add_argument("--max-tokens-in-batch", type=int, default=None,
                         help="Per-iteration scheduler token budget (vLLM's --max-num-batched-tokens / "
                              "SGLang's --chunked-prefill-size -- an aggregate cap shared across everything "
                              "packed into one scheduling step, not a per-request limit). Defaults to "
                              "max(prefill_tokens, 8192) when unset; pass the real server's actual value "
                              "here when known (check its startup logs/launch flags) for an accurate match.")
    parser.add_argument("--atten-input-file", default=None,
                         help="Path to a specific attention profiling CSV, overriding the default "
                              "./data/profiling/compute/<device>/<model>/attention.csv -- e.g. a block-size- "
                              "or backend-tagged variant like attention_combined_block16.csv. Must be the "
                              "*combined* file (standard + true-mixed rows merged), not the plain or "
                              "true-mixed-only ones -- the predictor loads one file and filters "
                              "is_true_mixed_batch rows out of it.")
    parser.add_argument("--output-dir", default="outputs/mi355x_deepseek_r1_validation")
    parser.add_argument("--log-dir", default="outputs/mi355x_deepseek_r1_validation/logs")
    parser.add_argument("--report", default="comparison_report.html")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    real_run = load_and_aggregate(args.run_dir)
    print(f"Loaded {real_run.n_runs} repetition(s) of the real benchmark from {real_run.source}", file=sys.stderr)
    topology = Topology(
        device=args.device,
        model_name=args.model_name,
        attn_tensor_parallel_size=args.attn_tp,
        attn_data_parallel_size=args.attn_dp,
        moe_tensor_parallel_size=args.moe_tp,
        moe_expert_parallel_size=args.moe_ep,
        num_pipeline_stages=args.pipeline_stages,
        num_replicas=args.num_replicas,
    )
    # real_run.benchmark exposes one canonical BenchmarkResult per concurrency level (workload
    # shape only -- num_prompts/prefill/decode lengths, identical across repetitions by
    # construction), so build_sweep works unchanged whether real_run came from one run or many.
    sim_points = build_sweep(
        real_run, topology, args.output_dir,
        cc_backend=args.cc_backend, network_device=args.network_device, request_mode=args.request_mode,
        block_size=args.block_size, enable_chunked_prefill=args.enable_chunked_prefill,
        max_tokens_in_batch=args.max_tokens_in_batch, atten_input_file=args.atten_input_file,
    )

    if args.dry_run:
        for sp in sim_points:
            print(f"# concurrency={sp.concurrency}")
            print(sp.command_str())
            print()
        return

    sim_results: List[SimResult] = []
    for sp in sim_points:
        print(f"Running concurrency={sp.concurrency} (run_id={sp.run_id}) ...", file=sys.stderr)
        sim_results.append(
            run_sim_point(sp, output_dir=args.output_dir, model_name=args.model_name, log_dir=Path(args.log_dir))
        )

    subtitle = (
        f"Real side: mean ± std across {real_run.n_runs} repeated benchmark run(s)"
        if real_run.n_runs > 1
        else None
    )

    # engine_label needs a sample BenchmarkResult alongside config: config.txt's bench_container
    # is missing entirely for bare .log captures, and has been found actively wrong for some
    # directory-based ones (see engine_label's docstring) -- the result block's own content is
    # the more reliable signal, when available.
    sample = real_run.benchmark[0] if real_run.benchmark else None
    engine = engine_label(real_run.config, sample)

    # Real-side request pattern: request_rate is parsed straight from the benchmark client's own
    # args (inf for closed-loop/max-concurrency sweeps, a finite target rate for open-loop).
    request_rate = sample.request_rate if sample is not None else None
    if request_rate is None or math.isinf(request_rate):
        loop_mode, request_rate_display = "closed-loop", "max-concurrency driven"
    else:
        loop_mode, request_rate_display = "open-loop", f"{request_rate:.3g} req/s"

    identity = RunIdentity(
        model_name=args.model_name,
        device=args.device,
        attn_tp=args.attn_tp,
        attn_dp=args.attn_dp,
        moe_tp=topology.moe_tensor_parallel_size,  # post_init-resolved, not the raw --moe-tp arg
        moe_ep=args.moe_ep,
        pipeline_stages=args.pipeline_stages,
        num_replicas=args.num_replicas,
        block_size=args.block_size,
        cc_backend=args.cc_backend,
        engine=engine,
        input_len=sample.random_input_len if sample is not None else None,
        output_len=sample.random_output_len if sample is not None else None,
        loop_mode=loop_mode,
        request_rate_display=request_rate_display,
    )

    write_html_report(
        real_run.results,
        sim_results,
        Path(args.report),
        title=f"{args.model_name} on {args.device}: real ({engine}) vs simulated",
        subtitle=subtitle,
        identity=identity,
    )
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
