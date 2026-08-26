"""Optuna HPO over Frontier's compute-calibration knobs (and cc-backend choice), minimizing
real-vs-simulated mismatch averaged across the validation scenarios we use regularly.

Why this is cheap per trial, not a full re-profiling run every time: the search space here is
restricted to `*_calibration_scale`/`*_batching_overhead_fraction` fields on
RandomForrestExecutionTimePredictorConfig, which are applied as a pure post-hoc multiply on an
already-trained/cached prediction (see sklearn_execution_time_predictor.py's
_get_calibration_scale and _get_model_hash/_get_prediction_cache_hash -- calibration values are
never part of the model-cache key). So every trial reuses the existing `cache/` directory's
trained models; only the discrete-event simulation itself re-runs per trial, not sklearn
training. cc_backend is included as a categorical choice since it materially changes
communication-cost prediction for TP>1 scenarios (it's a no-op for TP=1 ones, which always
short-circuit collective cost to 0 regardless of backend).

Disk safety (the whole reason this lives under tools/hpo/ instead of reusing run_validation.py
directly): each trial's raw frontier.main output (which includes a per-batch JSONL ledger that
alone runs tens of MB per concurrency point) is written to a throwaway directory under the
system temp dir, disabled down to just what system_metrics.json needs via
_ARTIFACT_DISABLE_FLAGS, and deleted immediately after its numbers are extracted -- win or lose.
Nothing under outputs/hpo/ is raw simulator output; it's the Optuna study database plus one
gzip-compressed JSONL line per trial (params + per-scenario per-metric errors + objective).

Parallelism (--n-jobs > 1): optuna.Study.optimize's n_jobs runs trials concurrently via a
thread pool *within one process*, not separate processes. That's fine for the actual work here
(each trial mostly blocks in subprocess.run, which releases the GIL), but it means two things
that would otherwise be silent hazards are handled explicitly below: (1) every worker thread
appends to the same trials.jsonl.gz -- guarded by _TRIALS_FILE_LOCK, since concurrent unlocked
writes could interleave and corrupt the gzip stream; (2) frontier's own sklearn/joblib code
happily oversubscribes all visible cores per process (the "leaked semlock" joblib/loky warnings
you may have seen come from exactly this) -- with n_jobs=6 that's 6 trials each independently
trying to use all 6 cores, thrashing far worse than clean 6-way parallelism. Each frontier.main
subprocess is launched with OMP/MKL/OPENBLAS/LOKY thread-count env vars pinned to 1 (_SUBPROCESS_ENV)
so the outer trial-level parallelism (what --n-jobs actually controls) is the only parallelism in
play, matching "use all N threads via N trials" rather than "N trials x each wanting all threads".

Usage:
    python3 -m tools.hpo.run_hpo --study-name deepseek_qwen3_calibration --n-trials 30
    python3 -m tools.hpo.run_hpo --study-name deepseek_qwen3_calibration --n-jobs 6  # unbounded, until stopped
    python3 -m tools.hpo.run_hpo --study-name deepseek_qwen3_calibration --n-trials 30 --resume
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import optuna

from tools.validation.compare_plots import _METRICS, _geo_mean_relative_error_pct
from tools.validation.frontier_cli_translator import Topology, build_sweep
from tools.validation.metrics_extractor import SimResult, extract_sim_result, find_run_dir
from tools.validation.real_log_aggregator import AggregatedResult, load_and_aggregate


# --- Scenarios ---------------------------------------------------------------------------------
# The two comparisons this session actually iterated on repeatedly: DeepSeek-R1 on mi355x
# (TP=8, MLA, exercises the collective_sim/analytical choice for real) and Qwen3-30B-A3B on h100
# (TP=1, dense/GQA MoE, cc_backend is a no-op here -- num_devices<=1 always short-circuits
# collective cost to 0 regardless of backend, confirmed in analytical_cc_backend.py and
# astra_sim_analytical_cc_backend.py). Deliberately maximizes diversity (hardware, attention
# family, topology) so a calibration that works well here has to generalize, not overfit one
# setup. Add more Scenario entries here later, per the task's "later we will use more models".


@dataclass(frozen=True)
class Scenario:
    name: str
    run_dir: str
    device: str
    model_name: str
    attn_tp: int
    moe_ep: int
    block_size: int
    atten_input_file: Optional[str] = None
    network_device: Optional[str] = None


SCENARIOS: List[Scenario] = [
    Scenario(
        name="deepseek_mi355x",
        run_dir="tools/inference_bench/deepseek/sglang/closed-loop",
        device="mi355x",
        model_name="deepseek-r1-0528",
        attn_tp=8,
        moe_ep=1,
        block_size=32,
        network_device="mi355x_8gpu",
    ),
    Scenario(
        name="qwen3_h100",
        run_dir="tools/inference_bench/Qwen3/sglang/sglang_combined.log",
        device="h100",
        model_name="qwen3-a3b-30b-moe",
        attn_tp=1,
        moe_ep=1,
        block_size=32,
        atten_input_file="data/profiling/compute/h100/qwen3-a3b-30b-moe/attention_combined.csv",
    ),
]

# --- Search space --------------------------------------------------------------------------
# First pass: the 9 always-on calibration scales (default 1.0, no interdependency validation)
# and the 2 batching-overhead fractions (default 0.1) on
# RandomForrestExecutionTimePredictorConfig -- confirmed cache-safe (see module docstring). The
# other ~17 base fields (prefill_phase_*/decode_phase_*/late_decode_*/low_prefill_*/
# high_prefill_* families) are Optional[None]-by-default and validated as all-or-nothing groups
# in BaseExecutionTimePredictorConfig.__post_init__ -- more complex to search safely, left for a
# later, smarter pass per the task.
_CALIBRATION_SCALE_FIELDS = [
    "attn_pre_proj_calibration_scale",
    "attn_post_proj_calibration_scale",
    "attn_decode_calibration_scale",
    "attn_kv_cache_save_calibration_scale",
    "mlp_up_proj_calibration_scale",
    "mlp_down_proj_calibration_scale",
    "moe_shuffling_calibration_scale",
    "moe_grouped_gemm_calibration_scale",
    "expert_parallel_communication_calibration_scale",
]
# Multiplicative correction -> log-uniform, consistent with how every ratio/error in this
# codebase's validation report is already reasoned about in log space (see
# compare_plots._geo_mean_relative_error_pct).
_CALIBRATION_SCALE_RANGE = (0.2, 5.0)

_BATCHING_OVERHEAD_FIELDS = [
    "attention_decode_batching_overhead_fraction",
    "attention_prefill_batching_overhead_fraction",
]
_BATCHING_OVERHEAD_RANGE = (0.0, 1.0)

# Top-level SimulationConfig field (frontier/config/quantization_manager.py's
# set_fp8_int8_approximation_scale), not one of RandomForrestExecutionTimePredictorConfig's --
# no `random_forrest_execution_time_predictor_config_` prefix, unlike the two families above.
# Only affects deepseek_mi355x in practice (confirmed: every DeepSeek log hits the FP8 profiling-
# precision-mismatch path -- attn_pre_proj/attn_post_proj/mlp_up_proj/mlp_down_proj/
# moe_grouped_gemm all real-precision FP8 vs. BF16-only profiling data; zero Qwen3 logs do,
# since its ops are profiled and served at the same precision) -- harmless to always include,
# since qwen3_h100 just never exercises the code path this scales.
_QUANTIZATION_FIELDS = ["fp8_int8_approximation_scale"]

# Only the two backends actually exercised (and confirmed functional) this session. vidur/
# aiconfigurator/astra_sim_analytical are plausible future additions, not included yet.
_CC_BACKEND_CHOICES = ["analytical", "collective_sim"]

# --- Artifact suppression, for disk safety (see module docstring) --------------------------
# store_request_metrics/store_token_completion_metrics are deliberately NOT here, despite
# looking like other "write this CSV or not" toggles: confirmed by reading metrics_store.py
# that they gate the underlying request-arrival/token-completion event RECORDING itself
# (on_request_arrival returns early when store_request_metrics is False), not just whether a
# CSV gets written -- disabling them silently zeroes out throughput_metrics/ttft_statistics/
# tpot_statistics/request_e2e_time_statistics too ("Missing arrival or completion data" /
# "No TTFT data available" in system_metrics.json, found by an HPO smoke test that returned
# all-None errors). The actual disk-heavy artifact is store_frontier_stage_batch_ledger (a
# per-batch JSONL trace, confirmed ~12-70MB per concurrency point) and its dependents below,
# all confirmed independent of summary-stat computation.
_ARTIFACT_DISABLE_FLAGS = [
    "--no-metrics_config_store_frontier_stage_batch_ledger",
    "--no-metrics_config_store_frontier_stage_batch_ledger_summary",
    "--no-metrics_config_store_batch_metrics",
    "--no-metrics_config_store_operation_metrics",
    "--no-metrics_config_store_utilization_metrics",
    "--no-metrics_config_store_plots",
    "--no-metrics_config_enable_chrome_trace",
    "--no-metrics_config_write_json_trace",
    "--no-metrics_config_enable_memory_time_series",
    "--no-metrics_config_enable_op_level_tracing",
    "--no-metrics_config_enable_per_layer_expansion",
]

# 30 min per concurrency point -- deliberately generous under --n-jobs>1: N trials contending for
# the same cores means each one's wall-clock time grows with N, and a spurious subprocess.run
# timeout is wasted work recorded as a fake "failure" rather than a real bad-calibration result.
_SIM_TIMEOUT_S = 1800

# See the module docstring's "Parallelism" section -- caps each frontier.main subprocess to a
# single core's worth of internal BLAS/joblib parallelism, so --n-jobs is the only place
# parallelism actually happens (otherwise N concurrent trials x each oversubscribing all cores
# internally thrashes far worse than clean N-way parallelism).
_SUBPROCESS_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "LOKY_MAX_CPU_COUNT": "1",
}

# Guards appends to trials.jsonl.gz: with --n-jobs>1, optuna runs trials on a thread pool within
# one process (not separate processes), so unsynchronized concurrent gzip appends from multiple
# worker threads could interleave mid-stream and corrupt the file.
_TRIALS_FILE_LOCK = threading.Lock()


def suggest_params(trial: "optuna.Trial") -> Dict[str, object]:
    params: Dict[str, object] = {
        "cc_backend": trial.suggest_categorical("cc_backend", _CC_BACKEND_CHOICES)
    }
    lo, hi = _CALIBRATION_SCALE_RANGE
    for field_name in _CALIBRATION_SCALE_FIELDS:
        params[field_name] = trial.suggest_float(field_name, lo, hi, log=True)
    lo, hi = _BATCHING_OVERHEAD_RANGE
    for field_name in _BATCHING_OVERHEAD_FIELDS:
        params[field_name] = trial.suggest_float(field_name, lo, hi)
    lo, hi = _CALIBRATION_SCALE_RANGE
    for field_name in _QUANTIZATION_FIELDS:
        params[field_name] = trial.suggest_float(field_name, lo, hi, log=True)
    return params


def _calibration_cli_flags(params: Dict[str, object]) -> List[str]:
    flags: List[str] = []
    for field_name in _CALIBRATION_SCALE_FIELDS + _BATCHING_OVERHEAD_FIELDS:
        flags.append(f"--random_forrest_execution_time_predictor_config_{field_name}")
        flags.append(f"{params[field_name]:.6f}")
    for field_name in _QUANTIZATION_FIELDS:
        flags.append(f"--{field_name}")
        flags.append(f"{params[field_name]:.6f}")
    return flags


def _real_value(real_results: Sequence[AggregatedResult], concurrency: int, field_name: str) -> Optional[float]:
    for r in real_results:
        if r.concurrency == concurrency:
            stats = r.get(field_name)
            return stats.mean if stats.n > 0 else None
    return None


def _log_tail(log_path: Path, n_lines: int = 60) -> str:
    """Last n_lines of a frontier.main log, for embedding in a failure record before its scratch
    dir gets deleted -- tracebacks are almost always near the end, and this stays small enough
    (a few KB) to not meaningfully bloat trials.jsonl.gz even across hundreds of failures."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {log_path}: {exc}>"
    return "\n".join(lines[-n_lines:])


def run_scenario(scenario: Scenario, params: Dict[str, object], trial_scratch_dir: Path) -> Dict[str, Optional[float]]:
    """Runs one scenario's full concurrency sweep under the trial's params, returns
    {metric_name: geometric-mean-abs-error-pct} for every metric in compare_plots._METRICS
    (None where too few points qualify -- see _geo_mean_relative_error_pct).
    """
    real_run = load_and_aggregate(scenario.run_dir)
    topology = Topology(
        device=scenario.device,
        model_name=scenario.model_name,
        attn_tensor_parallel_size=scenario.attn_tp,
        moe_expert_parallel_size=scenario.moe_ep,
    )
    sim_points = build_sweep(
        real_run, topology, str(trial_scratch_dir),
        cc_backend=params["cc_backend"],
        network_device=scenario.network_device,
        block_size=scenario.block_size,
        atten_input_file=scenario.atten_input_file,
    )

    extra_flags = _calibration_cli_flags(params) + _ARTIFACT_DISABLE_FLAGS
    log_dir = trial_scratch_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    sim_results: List[SimResult] = []
    for sp in sim_points:
        sp.args.extend(extra_flags)
        log_path = log_dir / f"{sp.run_id}.log"
        with open(log_path, "w") as f:
            result = subprocess.run(
                sp.command, stdout=f, stderr=subprocess.STDOUT, timeout=_SIM_TIMEOUT_S, env=_SUBPROCESS_ENV,
            )
        if result.returncode != 0:
            # Embed the actual failure directly in the exception, not just the (about-to-be-
            # deleted, since trial_scratch_dir is scratch) log path -- a real weekend-long run
            # pruned 91% of trials and every one of their tracebacks was lost with the scratch
            # dir, leaving nothing to diagnose after the fact. See _log_tail.
            tail = _log_tail(log_path)
            raise RuntimeError(
                f"frontier.main failed for {scenario.name}/{sp.run_id} "
                f"(returncode={result.returncode}):\n{tail}"
            )
        run_dir = find_run_dir(output_root=str(trial_scratch_dir), model_name=scenario.model_name, run_id=sp.run_id)
        sim_results.append(
            extract_sim_result(
                run_dir, run_id=sp.run_id, concurrency=sp.concurrency, num_prompts=sp.real.num_prompts,
            )
        )

    concurrencies = sorted({r.concurrency for r in real_run.results} | {s.concurrency for s in sim_results})
    sim_by_c = {s.concurrency: s for s in sim_results}

    errors: Dict[str, Optional[float]] = {}
    for name, real_field, sim_field, _unit in _METRICS:
        real_y = [_real_value(real_run.results, c, real_field) for c in concurrencies]
        sim_y = [getattr(sim_by_c.get(c), sim_field, None) for c in concurrencies]
        errors[name] = _geo_mean_relative_error_pct(real_y, sim_y)
    return errors


def objective_from_errors(per_scenario_errors: Dict[str, Dict[str, Optional[float]]]) -> Optional[float]:
    """Arithmetic mean of the per-metric abs errors, then arithmetic mean across scenarios --
    the simple starting objective the task asked for ("later we will configure something
    smarter"). Metrics/scenarios with no qualifying error (e.g. vLLM's missing E2E latency)
    are dropped from their mean rather than treated as 0 or infinity.
    """
    scenario_scores = []
    for errors in per_scenario_errors.values():
        values = [v for v in errors.values() if v is not None]
        if values:
            scenario_scores.append(sum(values) / len(values))
    if not scenario_scores:
        return None
    return sum(scenario_scores) / len(scenario_scores)


def make_objective(study_dir: Path):
    trials_path = study_dir / "trials.jsonl.gz"

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial)
        trial_scratch_dir = Path(tempfile.mkdtemp(prefix=f"frontier_hpo_trial{trial.number}_"))
        started = time.time()
        per_scenario_errors: Dict[str, Dict[str, Optional[float]]] = {}
        status = "ok"
        error_message = None
        try:
            for scenario in SCENARIOS:
                per_scenario_errors[scenario.name] = run_scenario(
                    scenario, params, trial_scratch_dir / scenario.name
                )
        except Exception as exc:  # noqa: BLE001 -- a bad param combo shouldn't kill the whole study
            status = "failed"
            error_message = str(exc)
        finally:
            shutil.rmtree(trial_scratch_dir, ignore_errors=True)

        objective_value = objective_from_errors(per_scenario_errors) if status == "ok" else None

        record = {
            "trial": trial.number,
            "status": status,
            "params": params,
            "per_scenario_errors": per_scenario_errors,
            "objective": objective_value,
            "duration_s": round(time.time() - started, 1),
            "error_message": error_message,
        }
        with _TRIALS_FILE_LOCK, gzip.open(trials_path, "at") as f:
            f.write(json.dumps(record) + "\n")

        if objective_value is None:
            raise optuna.TrialPruned(error_message or "no qualifying error pairs")
        return objective_value

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-name", required=True, help="Also the subdirectory name under outputs/hpo/")
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Cap on number of trials. Omit for unbounded -- run until --timeout-s elapses or you "
             "stop it yourself (Ctrl-C / tmux kill-session); already-completed trials stay in the "
             "study either way, resumable with --resume.",
    )
    parser.add_argument("--timeout-s", type=float, default=None, help="Overall wall-clock budget for the study")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Trials to run concurrently (thread pool within this one process -- see the module "
             "docstring's Parallelism section for why that's safe here and what it pins per "
             "subprocess). Match to available CPU cores, e.g. 6.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an existing study by name instead of failing if outputs/hpo/<study-name>/ already exists",
    )
    args = parser.parse_args()

    study_dir = Path("outputs/hpo") / args.study_name
    if study_dir.exists() and not args.resume:
        raise SystemExit(
            f"{study_dir} already exists -- pass --resume to add trials to it, or pick a new --study-name."
        )
    study_dir.mkdir(parents=True, exist_ok=True)

    db_path = study_dir / "study.db"
    # outputs/hpo/ is untracked by git (confirmed -- not even gitignored, just never added), so
    # nothing protects it from e.g. a stray `git clean -fd` or manual rm; combined with this box
    # being OOM-kill-prone under any real parallelism (confirmed via dmesg), a writer getting
    # SIGKILLed mid-transaction is a real, not hypothetical, risk to the one on-disk copy of a
    # study's entire history. Cheap insurance: snapshot before touching it, so --resume can never
    # start from a worse position than before you ran it, whatever silently ate the original.
    if db_path.exists():
        backup_path = study_dir / f"study.db.bak.{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(db_path, backup_path)
        print(f"Backed up existing study DB to {backup_path}", file=sys.stderr)

    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="minimize", load_if_exists=args.resume,
    )
    if args.resume and len(study.trials) == 0 and db_path.exists():
        print(
            f"WARNING: --resume was passed but the loaded study has 0 trials -- if you expected "
            f"an existing history, check {study_dir}/study.db.bak.* for a pre-existing snapshot "
            f"before continuing, since new trials will now start renumbering from 0.",
            file=sys.stderr,
        )

    def _periodic_backup(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        # Every 20 trials, not every trial -- shutil.copy2 of a growing sqlite file on every
        # single trial would add real I/O overhead for no real benefit at this cadence.
        if trial.number > 0 and trial.number % 20 == 0:
            shutil.copy2(db_path, study_dir / "study.db.bak.periodic")

    study.optimize(
        make_objective(study_dir), n_trials=args.n_trials, timeout=args.timeout_s, n_jobs=args.n_jobs,
        callbacks=[_periodic_backup],
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print(f"Best objective: {study.best_value:.2f}% (trial {study.best_trial.number})", file=sys.stderr)
        print(json.dumps(study.best_params, indent=2))
    else:
        # study.best_value/best_trial raise ValueError with no COMPLETE trials -- a real
        # outcome (not just an edge case), e.g. a --n-trials 1 run whose one draw got pruned.
        print(
            f"No completed trials yet ({len(study.trials)} run, 0 complete) -- "
            f"see {study_dir / 'trials.jsonl.gz'} for what happened to each.",
            file=sys.stderr,
        )
    print(f"Study DB: {study_dir / 'study.db'}", file=sys.stderr)
    print(f"Per-trial records: {study_dir / 'trials.jsonl.gz'}", file=sys.stderr)


if __name__ == "__main__":
    main()
