"""Static contract of the Qwen3 MI355X collection scripts (plan linear-op-recollection-contract-a, TASK-1 / TASK-3).

The scripts run only on the cluster, so their text is checked here: the linear_op stage must not force the numerically
wrong torch RoPE fallback (REQ-1), must run every method in LINEAR_PROFILE_METHODS into the same COLLECT_DIR (REQ-5), and
must pass the trace-keeping switch into the container.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN = REPO_ROOT / "profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch"


def _linear_stage() -> str:
    """The linear_op case block, comment lines removed (the comments mention the env var by name on purpose)."""
    text = MAIN.read_text()
    start = text.index("  linear_op)")
    end = text.index("  *) echo \"unknown STAGE", start)
    return "\n".join(line for line in text[start:end].splitlines() if not line.strip().startswith("#"))


def test_linear_stage_has_no_forced_rope_fallback():
    assert "FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK=1" not in _linear_stage()


def test_linear_stage_loops_profile_methods():
    stage = _linear_stage()
    assert re.search(r'for\s+M\s+in\s+\$\{LINEAR_PROFILE_METHODS:-cuda_event record_function\}', stage), stage
    assert '--profile_method "$M"' in stage or "--profile_method $M" in stage
    assert "--profile_method cuda_event" not in stage  # no hard-coded method left


def test_linear_stage_passes_trace_keep_switch_into_container():
    assert re.search(r'-e FRONTIER_RF_KEEP_TRACES(=|\b)', _linear_stage())
