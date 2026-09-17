"""Run one linear_op worker's task loop in a single process so rocprofv3 can trace it
(07_post_proj_rope_dip_root_cause.md, experiment X2). Same LinearOpWrapper.profile() path as
frontier/profiling/linear_op/main.py's worker; each task is wrapped in a roctx range "task_tp<TP>_tok<N>".

  rocprofv3 --kernel-trace --marker-trace --output-format csv -d <dir> -o tp2 -- \
      python profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/trace_ops.py --tp 2 --tokens 3000 4128 8000
"""
import argparse, json, os, sys, time

import torch

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.linear_op_wrapper import LinearOpWrapper
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan

p = argparse.ArgumentParser()
p.add_argument("--model", default="qwen3-a3b-30b-moe")
p.add_argument("--tp", type=int, required=True)
p.add_argument("--tokens", type=int, nargs="+", required=True)
p.add_argument("--output_dir", default="/tmp/trace_ops_out")
a = p.parse_args()

torch.cuda.set_device(0)
cfg = ModelConfig.from_model_name(a.model)
# same plan as the sweep: main.py --is_moe --num_tensor_parallel_workers 1 2 4 8 (FFN ops skipped, attention sharded ops only)
plan = build_profiling_plan(model_config=cfg, tp_size=a.tp, attn_tp=[1, 2, 4, 8], ffn_tp=[1, 2, 4, 8], is_moe=True)
print("enabled_ops:", plan["enabled_ops"], file=sys.stderr)
w = LinearOpWrapper(cfg, a.tp, "cuda_event", rank=0, output_dir=a.output_dir, profiling_plan=plan)
for tok in sorted(a.tokens, reverse=True):  # same descending order as the sweep
    torch.cuda.nvtx.range_push(f"task_tp{a.tp}_tok{tok}")
    r = w.profile(tok)
    torch.cuda.nvtx.range_pop()
    ts = r["time_stats"]
    rec = {"tp": a.tp, "tokens": tok, "t_wall": round(time.time(), 3), "W_ms": round(float(r["host_wall_per_forward_ms"]), 5),
           "backlog_ms": r["gpu_backlog_ms"], **{k: round(float(v["median"]), 5) for k, v in ts.items()}}
    print(json.dumps(rec), flush=True)
