"""Renders real-vs-simulated comparison plots across a concurrency sweep.

Purely visual comparison by design -- no pass/fail thresholds or scored verdicts.
Each chart is a single metric across concurrency levels, with two series (Real,
Simulated) so the viewer can judge similarity by eye.

The real side is one or more repeated benchmark runs (see real_log_aggregator), so it's
drawn as mean ± std across repetitions rather than a single point -- a lone real run is
noisy, and comparing a simulation against one noisy sample overstates how far off (or how
close) the simulation actually is. The simulated side stays a single line: Frontier's
simulation is deterministic given its inputs, so there's nothing to average.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.validation.metrics_extractor import SimResult
from tools.validation.real_log_aggregator import AggregatedResult, MetricStats, load_and_aggregate

# Fraction the real side's actual avg-tokens/request can diverge from the configured target
# before the token-context table flags it -- see RunIdentity / _token_context_table_html.
_LENGTH_MISMATCH_THRESHOLD = 0.03


@dataclass
class RunIdentity:
    """Everything the report's title + identity table need to name a comparison unambiguously.

    Assembled by the caller (run_validation.py), which already has all of this firsthand from
    its own CLI args / Topology / the real run's parsed config -- cheaper and more reliable than
    re-deriving it here from sim/real output files (e.g. engine identity in particular must come
    from log *content*, not a directory path -- see real_log_parser.engine_label's docstring for
    why the naive path-based approach is actively wrong for some real captures on disk).
    """

    model_name: str
    device: str
    attn_tp: int
    attn_dp: int
    moe_tp: int
    moe_ep: int
    pipeline_stages: int
    num_replicas: int
    block_size: int
    cc_backend: str
    engine: str  # "sglang" / "vLLM" / "unknown engine" -- see real_log_parser.engine_label
    input_len: Optional[int]
    output_len: Optional[int]
    loop_mode: str  # "closed-loop" or "open-loop"
    request_rate_display: str  # e.g. "max-concurrency driven (closed-loop)" or "4.32 req/s"


def _identity_table_html(identity: RunIdentity) -> str:
    """One-row "what exactly is being compared" table, rendered directly under the title."""
    topology = (
        f"PP{identity.pipeline_stages} x {identity.num_replicas} replica(s) | "
        f"AttnTP{identity.attn_tp} x AttnDP{identity.attn_dp} | "
        f"MoeTP{identity.moe_tp} x MoeEP{identity.moe_ep}"
    )
    rows = [
        ("Model", identity.model_name),
        ("Hardware", identity.device),
        ("Configuration", topology),
        ("Block size", str(identity.block_size)),
        ("CC backend", identity.cc_backend),
        ("Input / Output length", f"{_fmt(identity.input_len, '{:.0f}')} / {_fmt(identity.output_len, '{:.0f}')} tokens"),
        ("Request pattern", f"{identity.loop_mode} ({identity.request_rate_display})"),
        ("Engine (real side)", identity.engine),
    ]
    rows_html = "".join(f"<tr><th style='text-align:left'>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return (
        "<div style='overflow-x:auto'>"
        "<table cellpadding='6' style='border-collapse:collapse;font-family:sans-serif;font-size:13px'>"
        f"{rows_html}</table></div>"
    )


def _length_mismatch_note(
    real_stats: "MetricStats", successful_stats: "MetricStats", target_output_len: Optional[int]
) -> "tuple[str, bool]":
    """Real requests' actual avg output tokens vs. the configured target, as "actual / target
    (Δ%)", flagged when the divergence exceeds _LENGTH_MISMATCH_THRESHOLD.

    Frontier's simulator has no early-termination concept at all -- every simulated request
    generates exactly its configured decode length, always (confirmed: completed_requests ==
    total_requests in every existing sim output on disk, and request_num_decode_tokens always
    equals the configured decode_tokens with zero exceptions). So a mismatch here is entirely a
    real-side phenomenon (a request hit <eos> before the target length, or failed outright) --
    there is nothing to compute or compare on the sim side, which is why this note only covers
    the real columns.

    Returns (html_snippet, is_flagged) rather than raising/silently passing, so the caller can
    both render the value and decide whether to visually flag the cell -- same shape as the
    other _fmt_* helpers in this module.
    """
    if real_stats.n == 0 or successful_stats.n == 0 or not successful_stats.mean or target_output_len is None:
        return "n/a", False
    actual_avg = real_stats.mean / successful_stats.mean
    delta = (actual_avg / target_output_len) - 1
    flagged = abs(delta) > _LENGTH_MISMATCH_THRESHOLD
    text = f"{actual_avg:.1f} / {target_output_len} ({delta * 100:+.1f}%)"
    if flagged:
        text = f"⚠ {text}"
    return text, flagged


def _token_context_table_html(
    real: Sequence[AggregatedResult], sim: Sequence[SimResult], identity: RunIdentity
) -> str:
    """Point-in-context table: token volumes, request counts, and wall time per concurrency --
    the numbers the metric charts are ratios/rates *of*, plus the one sanity check that only
    makes sense against the real side (see _length_mismatch_note): did real requests actually
    reach their configured output length, on average, or are a meaningful fraction of them
    ending early (failed, or hit <eos> before the target)? A mismatch here explains part of any
    real-vs-sim divergence in the throughput/latency charts before jumping to "the simulator's
    execution-time model is wrong" -- the workloads being compared may not be quite the same
    shape after all.
    """
    concurrencies = sorted({r.concurrency for r in real} | {s.concurrency for s in sim})
    real_by_c = {r.concurrency: r for r in real}
    sim_by_c = {s.concurrency: s for s in sim}

    any_flagged = False
    rows_html = []
    for c in concurrencies:
        r, s = real_by_c.get(c), sim_by_c.get(c)
        real_in = r.get("total_input_tokens") if r else MetricStats()
        real_out = r.get("total_generated_tokens") if r else MetricStats()
        real_retok = r.get("total_generated_tokens_retokenized") if r else MetricStats()
        real_success = r.get("successful_requests") if r else MetricStats()
        real_dur = r.get("benchmark_duration_s") if r else MetricStats()
        mismatch_text, flagged = _length_mismatch_note(real_out, real_success, identity.output_len)
        any_flagged = any_flagged or flagged
        mismatch_style = " style='color:#b3261e;font-weight:bold'" if flagged else ""

        rows_html.append(
            "<tr>"
            f"<td>{c}</td>"
            f"<td>{_fmt_agg(real_in, '{:.0f}')}</td>"
            f"<td>{_fmt_agg(real_out, '{:.0f}')}</td>"
            f"<td>{_fmt_agg(real_retok, '{:.0f}')}</td>"
            f"<td{mismatch_style}>{mismatch_text}</td>"
            f"<td>{_fmt_agg(real_success, '{:.0f}')}</td>"
            f"<td>{_fmt_agg(real_dur, '{:.1f}')}</td>"
            f"<td>{_fmt(s.total_input_tokens if s else None, '{:.0f}')}</td>"
            f"<td>{_fmt(s.total_generated_tokens if s else None, '{:.0f}')}</td>"
            f"<td>{_fmt(s.successful_requests if s else None, '{:.0f}')}</td>"
            f"<td>{_fmt(s.benchmark_duration_s if s else None, '{:.1f}')}</td>"
            "</tr>"
        )

    warning_html = ""
    if any_flagged:
        warning_html = (
            "<p style='font-family:sans-serif;font-size:13px;color:#b3261e'>⚠ At least one "
            "concurrency level's real requests averaged output length more than "
            f"{_LENGTH_MISMATCH_THRESHOLD * 100:.0f}% away from the configured target "
            f"({identity.output_len} tokens) -- flagged rows below. This means the real and "
            "simulated workloads being compared are not quite the same shape (Frontier's "
            "simulator always generates exactly the configured length; the real requests here "
            "didn't), which is worth ruling out before attributing a throughput/latency "
            "mismatch to the execution-time model.</p>"
        )

    return (
        warning_html + "<div style='overflow-x:auto'>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse;font-family:sans-serif;"
        "font-size:13px;white-space:nowrap'>"
        "<thead><tr><th rowspan='2'>concurrency</th><th colspan='6'>Real</th><th colspan='4'>Simulated</th></tr>"
        "<tr>"
        "<th>Total input tok</th><th>Total output tok</th><th>Retokenized output tok</th>"
        "<th>Avg output tok/req (target)</th><th>Successful requests</th><th>Duration (s)</th>"
        "<th>Total input tok</th><th>Total output tok</th><th>Successful requests</th><th>Duration (s)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
        "<p style='font-family:sans-serif;font-size:12px;color:#777'>\"Retokenized output tok\" "
        "is only reported by sglang's bench_serving output (re-tokenizes the actual generated "
        "text to get an exact count) -- always \"n/a\" when the real side is vLLM. There is no "
        "sim-side equivalent to retokenization or to real-side request failures: the simulator "
        "never generates real text and never fails a request (see _length_mismatch_note).</p>"
    )

# Validated adjacent categorical pair (blue/orange) -- see dataviz skill references/palette.md.
_COLOR_REAL = "#2a78d6"
_COLOR_SIM = "#eb6834"

# Metrics grouped for display -- each group is exactly 3 metrics wide (throughput's natural
# grouping, matched for the latency groups) so the report renders 3-per-row with one row per
# group, group name as the row label. (group title, [(chart title, real field, sim field,
# y-axis label), ...]).
#
# Every field below already existed on both sides before this grouping was introduced --
# real_log_aggregator._AGGREGATE_FIELDS and metrics_extractor.SimResult already carried
# median/p99 for TTFT/TPOT and median for E2E; they just weren't wired into the report yet.
# ITL is deliberately not included here: Frontier has no true inter-token-gap statistic today
# (TPOT is a per-request average, not a distribution of individual token gaps) -- add it once
# that's real rather than fake it from TPOT.
_METRIC_GROUPS: List[tuple] = [
    ("Throughput", [
        ("Request throughput", "request_throughput_req_s", "request_throughput_req_s", "req/s"),
        ("Input token throughput", "input_token_throughput_tok_s", "input_token_throughput_tok_s", "tok/s"),
        ("Output token throughput", "output_token_throughput_tok_s", "output_token_throughput_tok_s", "tok/s"),
    ]),
    ("End-to-End Latency", [
        ("Mean E2E latency", "e2e_mean_ms", "e2e_mean_ms", "ms"),
        ("Median E2E latency", "e2e_median_ms", "e2e_median_ms", "ms"),
        ("P99 E2E latency", "e2e_p99_ms", "e2e_p99_ms", "ms"),
    ]),
    ("Time to First Token (TTFT)", [
        ("Mean TTFT", "ttft_mean_ms", "ttft_mean_ms", "ms"),
        ("Median TTFT", "ttft_median_ms", "ttft_median_ms", "ms"),
        ("P99 TTFT", "ttft_p99_ms", "ttft_p99_ms", "ms"),
    ]),
    ("Time per Output Token (TPOT)", [
        ("Mean TPOT", "tpot_mean_ms", "tpot_mean_ms", "ms"),
        ("Median TPOT", "tpot_median_ms", "tpot_median_ms", "ms"),
        ("P99 TPOT", "tpot_p99_ms", "tpot_p99_ms", "ms"),
    ]),
]
# Flat view for code that doesn't care about grouping (table columns, series computation).
_METRICS = [metric for _group, metrics in _METRIC_GROUPS for metric in metrics]


def _sim_series(records: Sequence[SimResult], concurrencies: List[int], field: str) -> List[Optional[float]]:
    by_conc = {r.concurrency: getattr(r, field, None) for r in records}
    return [by_conc.get(c) for c in concurrencies]


def _real_series_with_error(
    records: Sequence[AggregatedResult], concurrencies: List[int], field: str
) -> "tuple[List[Optional[float]], List[float]]":
    """Mean and std-dev (as a symmetric error-bar half-width) per concurrency level.

    A concurrency level with no repetition reporting this metric becomes a gap (None) rather
    than a fabricated zero -- matches how missing sim points are already handled.
    """
    by_conc = {r.concurrency: r.get(field) for r in records}
    means: List[Optional[float]] = []
    errors: List[float] = []
    for c in concurrencies:
        stats = by_conc.get(c, MetricStats())
        if stats.n == 0:
            means.append(None)
            errors.append(0.0)
        else:
            means.append(stats.mean)
            errors.append(stats.std or 0.0)
    return means, errors


# --- Real-vs-sim agreement statistics --------------------------------------------------------
#
# Two numbers summarize how well a metric's whole concurrency sweep agrees, on top of the
# eyeball comparison the plot already gives: a geometric-mean relative error (how far off, on
# average, and in which direction) and a Pearson correlation (whether the two series move
# together in shape, independent of any constant offset/scale between them). Both operate on
# per-concurrency (real_mean, sim_value) pairs built the same way the plotted series already are
# (_real_series_with_error / _sim_series), so "what counts as a data point here" never drifts
# from what's actually drawn on the chart.


def _relative_error_pct(real_value: Optional[float], sim_value: Optional[float]) -> Optional[float]:
    """Signed relative error of one sim value vs. its real counterpart, as a percentage.

    +12.3 means sim overpredicts real by 12.3%; -8.5 means sim underpredicts by 8.5%. None
    ("n/a" at render time) when either side is missing, or when either is zero/negative -- a
    ratio against a non-positive baseline isn't a meaningful percentage, and the log used by the
    geometric-mean aggregate below is undefined there too, so both stay consistent about which
    points count.
    """
    if real_value is None or sim_value is None or real_value <= 0 or sim_value <= 0:
        return None
    return (sim_value / real_value - 1) * 100


def _geo_mean_relative_error_pct(
    real_values: Sequence[Optional[float]], sim_values: Sequence[Optional[float]]
) -> Optional[float]:
    """Geometric-mean *absolute* relative error across paired (real, sim) values, as a percentage.

    This is a magnitude, not a signed bias: it answers "how far off is the simulator, on
    average" without direction cancelling out. Deliberately NOT a plain mean of the ratios --
    relative error is multiplicative, and the linear percentage scale it's usually expressed on
    is *asymmetric* around "no error" (sim at half of real is -50%; sim at 2x real is +100%, even
    though both are the "same size" miss just in opposite directions). Averaging in log space
    fixes that asymmetry, and taking the absolute value of each log-ratio before averaging is
    what keeps a +50%/-50% split from cancelling to a misleadingly reassuring ~0% (a plain signed
    geometric mean would average ln(1.5)=+0.405 and ln(0.667)=-0.405 to exactly zero, reporting
    "no error" for two horrible, oppositely-signed misses). Taking abs() first gives 0.405 both
    times, so the aggregate correctly reads as a ~50% average miss.

    This is a magnitude-only aggregate on purpose -- it does not tell you *which direction* the
    simulator tends to err in (a single number can't answer both "how big" and "which way" when
    the sign flips across concurrency levels; see _gsd, which is the one that stays signed and
    therefore *does* expose that kind of flip-flopping as inconsistency).

    Only strictly-positive, present-on-both-sides pairs contribute (same rule as
    _relative_error_pct, applied point-by-point here); returns None ("n/a") if no pair qualifies.
    """
    ratios = [
        sim / real
        for real, sim in zip(real_values, sim_values)
        if real is not None and sim is not None and real > 0 and sim > 0
    ]
    if not ratios:
        return None
    log_mean_abs = statistics.mean(abs(math.log(ratio)) for ratio in ratios)
    return (math.exp(log_mean_abs) - 1) * 100


def _gsd(
    real_values: Sequence[Optional[float]], sim_values: Sequence[Optional[float]]
) -> Optional[float]:
    """Geometric standard deviation of the sim/real ratio across paired values.

    Complements _geo_mean_relative_error_pct: that answers "how far off is the simulator, on
    average, and in which direction" (a single multiplicative bias). GSD answers a different
    question -- "how *consistent* is that multiplicative error across concurrency levels" --
    which neither the geometric mean nor the correlation below can tell you. Correlation only
    checks that Real and Sim move together in a straight line; it's blind to a constant
    multiplicative offset between them (Sim = 3.5 * Real at every point still gives r=1.0, a
    perfect *trend* match with a badly wrong *magnitude*). GSD is exp(sample-stdev(ln(ratio))):
    it comes out at exactly 1.0 when every point shares the same ratio -- however far that
    shared ratio is from 1 (a consistent bias, which the geometric-mean-error metric above
    already captures) -- and grows above 1.0 as the ratio itself varies from point to point, an
    operating-point-dependent error that a single averaged figure can hide entirely.

    Same positivity rule as _geo_mean_relative_error_pct (only strictly-positive, paired real/sim
    values contribute -- ln is undefined at/below zero: an explicit filter here, not a silent
    NaN/inf) and the same sample-statistics convention used throughout this module
    (statistics.stdev's default ddof=1, i.e. the n-1 sample standard deviation). A sample stdev
    needs at least 2 qualifying pairs to be defined at all; returns None ("n/a" at render time)
    below that, same as _pearson_r's handling of insufficient data.
    """
    ratios = [
        sim / real
        for real, sim in zip(real_values, sim_values)
        if real is not None and sim is not None and real > 0 and sim > 0
    ]
    if len(ratios) < 2:
        return None
    log_ratios = [math.log(ratio) for ratio in ratios]
    return math.exp(statistics.stdev(log_ratios))


def _pearson_r(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    """Pearson correlation coefficient between two paired series, hand-rolled.

    Deliberately not statistics.correlation: it's Python 3.10+ only, and this codebase already
    hand-rolls its small stats needs with statistics.mean/statistics.stdev rather than reach for
    version-gated stdlib additions or an extra dependency (see real_log_aggregator.MetricStats,
    which does the same for mean/stdev). Sample covariance and sample stdev both use the same
    (n-1) denominator, which cancels out of the ratio, so this is equivalent to the textbook
    Pearson r computed on population statistics.

    Unlike the geometric-mean error above, no positivity requirement -- correlation cares about
    whether the two series move together in shape, not about the sign or scale of their values.
    Needs at least 2 paired points and non-zero variance in *both* series (a single point has
    no spread to correlate against, and a constant series has no defined direction of
    relationship); otherwise returns None ("n/a" at render time) rather than raising or dividing
    by zero.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_std = statistics.stdev(x_vals)
    y_std = statistics.stdev(y_vals)
    if x_std == 0 or y_std == 0:
        return None
    x_mean = statistics.mean(x_vals)
    y_mean = statistics.mean(y_vals)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / (len(pairs) - 1)
    return covariance / (x_std * y_std)


def build_report(
    real: Sequence[AggregatedResult], sim: Sequence[SimResult], title: str = "Real vs Simulated"
) -> go.Figure:
    """One subplot per metric, x-axis = concurrency, two series (Real with error bars, Simulated) each.

    3 columns, one row per _METRIC_GROUPS entry (each group is exactly 3 metrics wide by
    construction), with the group name as that row's label down the left edge.
    """
    concurrencies = sorted({r.concurrency for r in real} | {s.concurrency for s in sim})
    cols = 3
    rows = len(_METRIC_GROUPS)

    # Each metric's real/sim series is computed once, up front, so the exact same pairing feeds
    # both the correlation/GSD shown in the subplot title (make_subplots needs all titles before
    # any trace can be added to a specific cell) and the traces plotted into that cell below.
    series_by_metric = []
    subplot_titles = []
    for name, real_field, sim_field, _unit in _METRICS:
        real_y, real_err = _real_series_with_error(real, concurrencies, real_field)
        sim_y = _sim_series(sim, concurrencies, sim_field)
        r = _pearson_r(real_y, sim_y)
        gsd_value = _gsd(real_y, sim_y)
        series_by_metric.append((real_y, real_err, sim_y))
        subplot_titles.append(
            f"{name} (r={_fmt_r(r)}, GSD={_fmt_gsd(gsd_value)})"
        )

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=subplot_titles,
        row_titles=[group_title for group_title, _metrics in _METRIC_GROUPS],
    )

    for i, (_name, _real_field, _sim_field, unit) in enumerate(_METRICS):
        row, col = i // cols + 1, i % cols + 1
        real_y, real_err, sim_y = series_by_metric[i]

        fig.add_trace(
            go.Scatter(
                x=concurrencies, y=real_y, name="Real", legendgroup="real",
                showlegend=(i == 0), mode="lines+markers",
                line=dict(color=_COLOR_REAL, width=2), marker=dict(size=8),
                error_y=dict(type="data", array=real_err, visible=True, color=_COLOR_REAL, thickness=1.5, width=4),
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=concurrencies, y=sim_y, name="Simulated", legendgroup="sim",
                showlegend=(i == 0), mode="lines+markers",
                line=dict(color=_COLOR_SIM, width=2, dash="dot"), marker=dict(size=8, symbol="diamond"),
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="concurrency", type="log", row=row, col=col)
        fig.update_yaxes(title_text=unit, row=row, col=col)

    fig.update_layout(
        title=title,
        height=320 * rows,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fmt(value: Optional[float], spec: str = "{:.2f}") -> str:
    """Format a plain (sim-side) metric value, or a placeholder when it's missing."""
    return spec.format(value) if value is not None else "n/a"


def _fmt_agg(stats: MetricStats, spec: str = "{:.2f}") -> str:
    """Format a real-side aggregated metric: bare value at n=1, "mean ± std (n=N)" at n>1.

    vLLM's bench_serving output also omits E2E latency and achieved-concurrency entirely unless
    --percentile-metrics includes "e2el", so n==0 (no repetition reported this field) is
    routinely hit too -- unlike sglang, where every field here is always populated.
    """
    if stats.n == 0:
        return "n/a"
    if stats.n == 1:
        return spec.format(stats.mean)
    return f"{spec.format(stats.mean)} ± {spec.format(stats.std)} (n={stats.n})"


def _fmt_delta_pct(value: Optional[float]) -> str:
    """Format a signed per-point relative-error percentage, e.g. "+12.3%" / "-8.5%", or "n/a"
    when it couldn't be computed (see _relative_error_pct for why)."""
    return f"{value:+.1f}%" if value is not None else "n/a"


def _fmt_abs_pct(value: Optional[float]) -> str:
    """Format the geometric-mean *absolute* relative-error aggregate, e.g. "50.0%" -- no forced
    sign, since this is a magnitude (see _geo_mean_relative_error_pct), not a signed bias; a "+"
    here would wrongly imply the simulator always overpredicts."""
    return f"{value:.1f}%" if value is not None else "n/a"


def _fmt_r(value: Optional[float]) -> str:
    """Format a Pearson correlation coefficient to 2 decimals, or "n/a" (see _pearson_r)."""
    return f"{value:.2f}" if value is not None else "n/a"


def _fmt_gsd(value: Optional[float]) -> str:
    """Format a geometric standard deviation as a multiplicative spread factor, e.g. "1.15×"
    (1.0x = perfectly consistent error across concurrency levels), or "n/a" (see _gsd)."""
    return f"{value:.2f}×" if value is not None else "n/a"


# One decimal is enough resolution for millisecond latencies; everything else (throughputs)
# gets two -- matches the precision the previous hand-written table used for the 4 metrics it
# covered, now applied uniformly across all of _METRICS.
_MS_FIELDS = {
    "e2e_mean_ms", "e2e_median_ms", "e2e_p99_ms",
    "ttft_mean_ms", "ttft_median_ms", "ttft_p99_ms",
    "tpot_mean_ms", "tpot_median_ms", "tpot_p99_ms",
}


def _table_spec(real_field: str) -> str:
    return "{:.1f}" if real_field in _MS_FIELDS else "{:.2f}"


def _data_table_html(real: Sequence[AggregatedResult], sim: Sequence[SimResult]) -> str:
    """Plain accessibility-fallback table -- reference data, not a verdict.

    Covers every metric in _METRICS (so the table and the plot never drift apart), three
    sub-columns each: Real | Sim | Δ, where Δ is the signed per-point relative error from
    _relative_error_pct. Three summary rows follow the per-concurrency rows: a per-metric
    geometric-mean relative error (average multiplicative bias), a per-metric geometric standard
    deviation (consistency of that bias across concurrency levels -- see _gsd for why this is a
    distinct question from both the error and the correlation below), and a per-metric
    real-vs-sim Pearson correlation (whether the two series share a trend, independent of any
    constant offset/scale between them). The error and correlation aggregates are also annotated
    onto the plot's subplot titles; all three are given here as an at-a-glance table for whoever's
    scanning the HTML fallback (or reading it after the JS-rendered chart has been stripped, e.g.
    in a text-only diff of the report).
    """
    concurrencies = sorted({r.concurrency for r in real} | {s.concurrency for s in sim})
    real_by_c = {r.concurrency: r for r in real}
    sim_by_c = {s.concurrency: s for s in sim}
    n_metrics = len(_METRICS)

    rows_html = []
    for c in concurrencies:
        r, s = real_by_c.get(c), sim_by_c.get(c)
        if r is None or s is None:
            rows_html.append(
                f"<tr><td>{c}</td>"
                f"<td colspan='{3 * n_metrics}'>missing data for this concurrency level</td></tr>"
            )
            continue
        cells = [f"<td>{c}</td>"]
        for _name, real_field, sim_field, _unit in _METRICS:
            spec = _table_spec(real_field)
            real_stats = r.get(real_field)
            real_value = real_stats.mean if real_stats.n > 0 else None
            sim_value = getattr(s, sim_field, None)
            delta = _relative_error_pct(real_value, sim_value)
            cells.append(f"<td>{_fmt_agg(real_stats, spec)}</td>")
            cells.append(f"<td>{_fmt(sim_value, spec)}</td>")
            cells.append(f"<td>{_fmt_delta_pct(delta)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    # Summary rows: one aggregate figure per metric, spanning that metric's Real|Sim|Δ group so
    # it reads as a single value under the 3-wide column band rather than three empty-looking
    # cells. Computed over the same per-concurrency (real_mean, sim_value) pairing as the rows
    # above (via _real_series_with_error / _sim_series), not just the concurrencies where a full
    # row happened to render -- a metric can be n/a in the table's colspan-fallback row (missing
    # from one side entirely) yet still have a usable point at another concurrency level.
    geo_cells = ["<td><strong>Geometric mean abs. error</strong></td>"]
    gsd_cells = ["<td><strong>GSD</strong></td>"]
    corr_cells = ["<td><strong>Correlation (r)</strong></td>"]
    for _name, real_field, sim_field, _unit in _METRICS:
        real_y, _real_err = _real_series_with_error(real, concurrencies, real_field)
        sim_y = _sim_series(sim, concurrencies, sim_field)
        geo_mean_pct = _geo_mean_relative_error_pct(real_y, sim_y)
        gsd_value = _gsd(real_y, sim_y)
        r_value = _pearson_r(real_y, sim_y)
        geo_cells.append(
            f"<td colspan='3' style='text-align:center'>{_fmt_abs_pct(geo_mean_pct)}</td>"
        )
        gsd_cells.append(f"<td colspan='3' style='text-align:center'>{_fmt_gsd(gsd_value)}</td>")
        corr_cells.append(f"<td colspan='3' style='text-align:center'>{_fmt_r(r_value)}</td>")
    rows_html.append("<tr>" + "".join(geo_cells) + "</tr>")
    rows_html.append("<tr>" + "".join(gsd_cells) + "</tr>")
    rows_html.append("<tr>" + "".join(corr_cells) + "</tr>")

    # Three header rows: group name (spans that group's 3 metrics x 3 subcols = 9), metric name
    # (spans its own Real|Sim|Δ = 3), then the Real/Sim/Δ subcols themselves.
    header_group_row = "".join(
        f"<th colspan='{3 * len(metrics)}'>{group_title}</th>" for group_title, metrics in _METRIC_GROUPS
    )
    header_metric_row = "".join(f"<th colspan='3'>{name} ({unit})</th>" for name, _rf, _sf, unit in _METRICS)
    header_subcols = "<th>Real</th><th>Sim</th><th>Δ</th>" * n_metrics

    return (
        "<p style='font-family:sans-serif;font-size:13px;color:#555'>Real columns show "
        "mean ± std across repeated runs of the same benchmark config (n shown when &gt;1; "
        "a bare value means only one repetition covered that field). vLLM's bench_serving "
        "output never reports End-to-End latency at all (confirmed across every vLLM capture "
        "in this dataset, not just a --percentile-metrics omission), so that whole group reads "
        "\"n/a\" when the real side is vLLM -- a capability gap in the benchmark tool, not a "
        "validation failure. Δ is the signed relative error of Sim vs. Real at that point ("
        "<code>(sim/real - 1) * 100</code>, e.g. +12.3% means sim overpredicts real by 12.3%); "
        "it's \"n/a\" wherever either side is missing or non-positive. The \"Geometric mean "
        "abs. error\" summary row is the per-metric aggregate <em>magnitude</em> of that error "
        "across all concurrency levels -- a geometric mean of |ln(sim/real)| exponentiated back "
        "to a percentage, so a run that's +50% at one concurrency and -50% at another reports "
        "~50% here instead of misleadingly cancelling to ~0% (see _geo_mean_relative_error_pct). "
        "It never carries a sign, since it deliberately no longer says which direction the error "
        "runs -- the per-point Δ cells above already show that at each concurrency level. The "
        "\"GSD\" (geometric standard deviation) row measures how <em>consistent</em> that error "
        "is, sign included, across concurrency levels, as a spread factor (1.00× = the signed "
        "sim/real ratio is identical at every concurrency level, however far from 1 that ratio "
        "is; larger values mean the ratio itself varies from point to point -- including "
        "flipping between over- and under-prediction) -- a simulator can have a small average "
        "error and a large GSD (accurate on average, but only by cancellation) or vice versa. "
        "The \"Correlation (r)\" row is the Pearson correlation between the real and simulated "
        "series across concurrency levels for that metric, independent of any constant offset "
        "or scale between them -- note this means high correlation alone does not imply "
        "agreement (e.g. Sim = 3.5 × Real at every point still gives r=1.00); use it alongside "
        "the error and GSD rows above, not in place of them.</p>"
        "<div style='overflow-x:auto'>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse;font-family:sans-serif;"
        "font-size:13px;white-space:nowrap'>"
        f"<thead><tr><th rowspan='3'>concurrency</th>{header_group_row}</tr>"
        f"<tr>{header_metric_row}</tr>"
        f"<tr>{header_subcols}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
    )


def write_html_report(
    real: Sequence[AggregatedResult],
    sim: Sequence[SimResult],
    output_path: Path,
    title: str,
    subtitle: Optional[str] = None,
    identity: Optional[RunIdentity] = None,
) -> None:
    """identity is optional (compare_plots.main()'s bare CLI has no Topology/RunConfig to build
    one from) -- when given, it adds the identity table and the token/request context table
    (which needs identity.output_len for its length-mismatch check) on top of the plot and data
    table that render either way."""
    fig = build_report(real, sim, title)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    identity_html = f"<h2>Run identity</h2>{_identity_table_html(identity)}" if identity else ""
    context_html = (
        f"<h2>Request / token context</h2>{_token_context_table_html(real, sim, identity)}"
        if identity else ""
    )
    table_html = _data_table_html(real, sim)
    subtitle_html = f"<p style='color:#555'>{subtitle}</p>" if subtitle else ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
        f"<body style='font-family:sans-serif;max-width:1400px;margin:2rem auto'>"
        f"<h1>{title}</h1>"
        f"{subtitle_html}"
        f"{identity_html}"
        f"{context_html}"
        f"{fig_html}"
        f"<h2>Data</h2>{table_html}"
        f"</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-dir", required=True,
        help="Real benchmark_results/run_<label>/ directory, or a directory containing several "
        "repeated run_<label>/ subdirectories (same config, different reps) -- see real_log_aggregator",
    )
    parser.add_argument("--sim-json", required=True, help="JSON list of SimResult records (one per concurrency level, from metrics_extractor --json)")
    parser.add_argument("--title", default="Real vs Simulated")
    parser.add_argument("-o", "--output", default="comparison_report.html")
    args = parser.parse_args()

    agg_run = load_and_aggregate(args.real_dir)

    sim_payload = json.loads(Path(args.sim_json).read_text())
    sim_records = [SimResult(**s) for s in sim_payload]

    subtitle = f"Real side: mean ± std across {agg_run.n_runs} repeated benchmark run(s)" if agg_run.n_runs > 1 else None
    write_html_report(agg_run.results, sim_records, Path(args.output), args.title, subtitle=subtitle)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
