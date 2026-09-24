"""Numeric check of the torch RoPE fallback in frontier/.../rotary_embedding.py against a reference neox RoPE (CPU)."""
import sys, torch
sys.path.insert(0, "/home/dn/Frontier-qwen3-profiling")
import importlib.util
spec = importlib.util.spec_from_file_location("re_mod", "/home/dn/Frontier-qwen3-profiling/frontier/profiling/common/layers/rotary_embedding.py")
# the module imports TimerStatsStore (fine on CPU) and nothing GPU-specific at import time
re_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(re_mod)
torch.manual_seed(0)
head_dim, base, max_pos = 128, 10_000_000, 4096
def reference_neox(q, positions, heads):
    # vLLM neox-style: per head, pairs (i, i+64), inv_freq over head_dim/2 = 64 frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    freqs = positions.to(torch.float64)[:, None] * inv_freq[None, :]            # [N, 64]
    cos, sin = freqs.cos(), freqs.sin()
    x = q.to(torch.float64).view(q.shape[0], heads, head_dim)
    x1, x2 = x[..., :64], x[..., 64:]
    out = torch.cat([x1 * cos[:, None, :] - x2 * sin[:, None, :], x2 * cos[:, None, :] + x1 * sin[:, None, :]], dim=-1)
    return out.view(q.shape)
def fallback(q, positions):
    # exactly what RotaryEmbedding.forward does on the fallback path, rebuilt on CPU (the class builds its cache on "cuda")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float)
    cache = torch.cat([torch.einsum("i,j->ij", t, inv_freq).cos(), torch.einsum("i,j->ij", t, inv_freq).sin()], dim=-1)  # [max_pos, 128]
    cos_sin = cache.index_select(0, positions); cos, sin = cos_sin.chunk(2, dim=-1)                                      # [N,64] each
    qq, kk = re_mod._apply_rotary_pos_emb(q.clone(), q.clone(), cos, sin, is_neox_style=True)
    return qq, cos.shape[-1]
for tp in [1, 2, 4, 8]:
    heads = 32 // tp; N = 64
    q = torch.randn(N, heads * head_dim, dtype=torch.float32)
    positions = torch.arange(1, N + 1)  # position 0 excluded (cos=1, sin=0 there)
    fb, rotary_dim_seen = fallback(q, positions)
    ref = reference_neox(q, positions, heads).to(torch.float32)
    untouched = (fb == q).float().mean().item()
    pred_untouched = 1 - 64 / (heads * head_dim)
    diff_ref = (fb - ref).abs()
    head0_cols = slice(0, 128)
    print(f"TP{tp} heads={heads}: rotary_dim seen by fallback={rotary_dim_seen} | fraction of q elements bit-identical to input: "
          f"{untouched:.4f} (predicted {pred_untouched:.4f}) | max|fallback-ref|: all={diff_ref.max():.3f}, head0={diff_ref[:, head0_cols].max():.3f}, "
          f"other heads={diff_ref[:, 128:].max() if heads > 1 else float('nan'):.3f} | fraction of elements where fallback==ref: {(diff_ref < 1e-5).float().mean():.4f}")
# position 0 sanity: both should be identity
q = torch.randn(1, 2048); fb, _ = fallback(q, torch.tensor([0])); print("position 0: fallback==input:", torch.equal(fb, q), "| ref==input:", torch.allclose(reference_neox(q, torch.tensor([0]), 16).float(), q, atol=1e-6))
