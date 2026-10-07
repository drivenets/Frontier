"""Repeatable read-only Dataset A inventory; no simulator imports."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def audit(root):
    root = Path(root)
    linear_path = root / "data/local_datasets/mi355x/qwen3-a3b-30b-moe/linear_op/2026-09-23_0949_dense_workbacklog_settle8/linear_op.csv"
    attention_path = root / "data/profiling/compute/mi355x/qwen3-a3b-30b-moe/attention_combined.csv"
    lin = pd.read_csv(linear_path, low_memory=False)
    att = pd.read_csv(attention_path, low_memory=False)
    ops = ['attn_pre_proj','attn_post_proj','attn_rope','input_layernorm','post_attention_layernorm','emb']
    features = ['decode_batch_size','decode_avg_kv_cache_size','num_prefill_seqs','total_prefill_tokens',
                'total_batch_size','batch_composition_ratio','total_tokens']
    mixed = att[att['mode']=='true_mixed']
    transitions = {}
    for tp in [1,4]:
        values = lin[lin['num_tensor_parallel_workers']==tp].set_index('num_tokens')['time_stats.attn_post_proj.median']
        transitions[str(tp)] = dict(before_ms=float(values[8192]), after_ms=float(values[8208]),
                                    increase_percent=float((values[8208]/values[8192]-1)*100))
    return dict(linear=dict(path=str(linear_path),md5=hashlib.md5(linear_path.read_bytes()).hexdigest(),shape=list(lin.shape),
                           targets_by_tp={op:{str(tp):int(n) for tp,n in lin.groupby('num_tensor_parallel_workers')['time_stats.'+op+'.median'].count().items()} for op in ops},
                           post_projection_8192_to_8208=transitions),
                attention=dict(path=str(attention_path),sha256=hashlib.sha256(attention_path.read_bytes()).hexdigest(),
                               rows=len(att),regimes={str(k):int(v) for k,v in att['mode'].value_counts().items()},
                               by_page_tp_regime=[dict(block_size=int(k[0]),tp=int(k[1]),mode=k[2],rows=int(v)) for k,v in att.groupby(['block_size','num_tensor_parallel_workers','mode']).size().items()],
                               mixed_complete_decode_features=int(mixed[features].notna().all(axis=1).sum()),
                               mixed_targets={op:int(mixed['time_stats.'+op+'.median'].notna().sum()) for op in ['attn_prefill','attn_decode','attn_kv_cache_save']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[3]))
    p.add_argument('--output',required=True)
    a=p.parse_args()
    with open(a.output,'x') as f:
        json.dump(audit(a.repo),f,indent=2,sort_keys=True)
        f.write('\n')
