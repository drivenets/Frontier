"""Numerics of the RoPE paths used by linear_op profiling (plan linear-op-recollection-contract-a, TASK-1 / REQ-1).

Reference: neox-style rotary embedding applied PER HEAD over the full head_size — pairs (i, i + head_size/2) with
inv_freq = base ** (-2i / rotary_dim), computed in float64 (the reference first used in the 2026-09-17 session-scratchpad check
that showed the pre-fix fallback rotating only the first rotary_dim/2 columns of head 0 of the flattened [N, heads*head_size] tensor;
this file is now its only home).

- CPU: the fixed torch fallback (`_apply_rotary_pos_emb(..., head_size=...)`) must match the reference on every head, both
  neox and GPT-J (interleaved) pairing, in float32 to 1e-3 and in bf16 to 2e-2 (4e-2 for the GPU kernel test: positions up to
  4096, bf16 end to end). The float32 bound is set by the production cache: cos/sin of angles up to
  ~4095 rad computed in float32 carry ~4e-4 absolute error (measured 4.1e-4), and bf16 has 8 mantissa bits (|x| up to ~4 for
  unit-normal inputs -> ~1.6e-2). The plan's "1e-2 (bf16)" is therefore split into float32 1e-3 and bf16 2e-2 - recorded as a
  deviation; the pre-fix bug produced O(1) errors on every head but head 0, far above either bound.
- `rope_impl_name(rotary_emb)` classifies the object the model actually calls: our class + fused vllm._custom_ops kernel ->
  "vllm_kernel" (the kernel family SGLang launches on this image), our class on the torch path -> "torch_fallback", a vLLM
  object -> "vllm_object_<forward method>" (on ROCm vLLM 0.9.2 dispatches to its 17-kernel torch path, never the fused op).
- `get_rope` prefers our class whenever the fused op imports.
- GPU (skipped without CUDA + vLLM): with the fallback env unset, `get_rope` returns Frontier's RotaryEmbedding on the fused
  vllm._custom_ops path (NOT vLLM's object), `rope_impl_name` reports "vllm_kernel", and the kernel output matches the reference.
"""
import math
import os

import pytest

torch = pytest.importorskip("torch")

from frontier.profiling.common.layers import rotary_embedding as re_mod  # noqa: E402

HEAD_SIZE, BASE, MAX_POS = 128, 10_000_000.0, 4096


def _inv_freq(rotary_dim: int, dtype=torch.float64) -> torch.Tensor:
    return 1.0 / (BASE ** (torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim))


def reference_rope(x: torch.Tensor, positions: torch.Tensor, heads: int, *, is_neox_style: bool) -> torch.Tensor:
    """float64 reference over the full head_size for every head; x is [N, heads*HEAD_SIZE]."""
    freqs = positions.to(torch.float64)[:, None] * _inv_freq(HEAD_SIZE)[None, :]  # [N, 64]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    xh = x.to(torch.float64).view(x.shape[0], heads, HEAD_SIZE)
    if is_neox_style:
        x1, x2 = xh[..., : HEAD_SIZE // 2], xh[..., HEAD_SIZE // 2 :]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1, x2 = xh[..., 0::2], xh[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)
    return out.view(x.shape)


def _fallback_cos_sin(positions: torch.Tensor, dtype: torch.dtype):
    """Exactly what RotaryEmbedding.forward feeds the fallback: cache = cat(cos, sin) over rotary_dim/2 freqs, chunked."""
    freqs = positions.to(torch.float32)[:, None] * _inv_freq(HEAD_SIZE, torch.float32)[None, :]
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype)
    return cache.chunk(2, dim=-1)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("is_neox_style", [True, False], ids=["neox", "gptj"])
def test_fallback_matches_reference_per_head(tp: int, is_neox_style: bool):
    torch.manual_seed(0)
    heads, n = 32 // tp, 96
    positions = torch.cat([torch.tensor([0, 1, 2]), torch.randint(3, MAX_POS, (n - 3,))])
    q = torch.randn(n, heads * HEAD_SIZE, dtype=torch.float32)
    k = torch.randn(n, (heads // 8 or 1) * HEAD_SIZE, dtype=torch.float32)
    cos, sin = _fallback_cos_sin(positions, torch.float32)
    q_out, k_out = re_mod._apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin, is_neox_style, head_size=HEAD_SIZE)
    q_ref = reference_rope(q, positions, heads, is_neox_style=is_neox_style).to(torch.float32)
    k_ref = reference_rope(k, positions, heads // 8 or 1, is_neox_style=is_neox_style).to(torch.float32)
    assert (q_out - q_ref).abs().max().item() <= 1e-3
    assert (k_out - k_ref).abs().max().item() <= 1e-3
    # every head is rotated: at positions >= 1 essentially no element survives bit-identical (the old bug left 1 - 64/(heads*128))
    untouched = (q_out[3:] == q[3:]).float().mean().item()
    assert untouched < 0.01, f"fraction of q left untouched: {untouched}"


def test_fallback_bf16_within_tolerance():
    torch.manual_seed(1)
    heads, n = 8, 64
    positions = torch.randint(1, MAX_POS, (n,))
    q = torch.randn(n, heads * HEAD_SIZE).to(torch.bfloat16)
    cos, sin = _fallback_cos_sin(positions, torch.bfloat16)
    q_out, _ = re_mod._apply_rotary_pos_emb(q.clone(), q.clone(), cos, sin, True, head_size=HEAD_SIZE)
    q_ref = reference_rope(q.to(torch.float32), positions, heads, is_neox_style=True)
    assert (q_out.to(torch.float64) - q_ref).abs().max().item() <= 2e-2


def test_fallback_position_zero_is_identity():
    q = torch.randn(4, 16 * HEAD_SIZE)
    cos, sin = _fallback_cos_sin(torch.zeros(4, dtype=torch.long), torch.float32)
    q_out, _ = re_mod._apply_rotary_pos_emb(q.clone(), q.clone(), cos, sin, True, head_size=HEAD_SIZE)
    assert torch.equal(q_out, q)


def test_fallback_partial_rotary_dim_leaves_pass_through_columns():
    """rotary_dim < head_size (cos width = rotary_dim/2): only the first rotary_dim columns OF EACH HEAD rotate."""
    torch.manual_seed(2)
    heads, n, rotary_dim = 4, 8, 64
    positions = torch.randint(1, MAX_POS, (n,))
    freqs = positions.to(torch.float32)[:, None] * _inv_freq(rotary_dim, torch.float32)[None, :]
    cos, sin = freqs.cos(), freqs.sin()  # [n, 32]
    q = torch.randn(n, heads * HEAD_SIZE)
    q_out, _ = re_mod._apply_rotary_pos_emb(q.clone(), q.clone(), cos, sin, True, head_size=HEAD_SIZE)
    qh, oh = q.view(n, heads, HEAD_SIZE), q_out.view(n, heads, HEAD_SIZE)
    assert torch.equal(oh[..., rotary_dim:], qh[..., rotary_dim:])  # pass-through per head
    assert not torch.equal(oh[..., :rotary_dim], qh[..., :rotary_dim])
    assert not torch.equal(oh[:, 1, :rotary_dim], qh[:, 1, :rotary_dim])  # head 1 rotated too


def test_fallback_rejects_rotary_dim_wider_than_head():
    cos = torch.zeros(2, HEAD_SIZE)  # cos width HEAD_SIZE -> rotary_dim 2*HEAD_SIZE > head_size
    with pytest.raises(ValueError, match="exceeds head_size"):
        re_mod._apply_rotary_pos_emb(torch.zeros(2, HEAD_SIZE), torch.zeros(2, HEAD_SIZE), cos, cos, True, head_size=HEAD_SIZE)


def test_rope_impl_name_unknown_for_foreign_objects():
    assert re_mod.rope_impl_name(torch.nn.Identity()) == "unknown"


class _FrontierLike(re_mod.RotaryEmbedding):
    def __init__(self):  # no cache build (it would need a GPU)
        torch.nn.Module.__init__(self)


def test_rope_impl_name_from_object(monkeypatch):
    class _VllmLike:
        def forward_native(self):
            pass
    _VllmLike.__module__ = "vllm.model_executor.layers.rotary_embedding.base"
    v = _VllmLike(); v._forward_method = v.forward_native

    monkeypatch.setattr(re_mod, "_fused_rope_kernel_available", lambda: True)
    assert re_mod.rope_impl_name(_FrontierLike()) == "vllm_kernel"      # our class + fused custom op
    monkeypatch.setattr(re_mod, "_fused_rope_kernel_available", lambda: False)
    assert re_mod.rope_impl_name(_FrontierLike()) == "torch_fallback"   # our class on the torch path
    assert re_mod.rope_impl_name(v) == "vllm_object_forward_native"     # vLLM's object: dispatch is not the fused op on ROCm
    assert re_mod.rope_impl_name(None) == "none"


def test_fused_kernel_availability_follows_env_and_import(monkeypatch):
    monkeypatch.setattr(re_mod, "_load_vllm_custom_ops", lambda: object())
    monkeypatch.delenv("FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK", raising=False)
    monkeypatch.delenv("FRONTIER_PROFILING_FORCE_TORCH_FALLBACK", raising=False)
    monkeypatch.setattr(re_mod, "TimerStatsStore", lambda: type("S", (), {"disabled": False})())
    assert re_mod._fused_rope_kernel_available() is True
    monkeypatch.setenv("FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK", "1")
    assert re_mod._fused_rope_kernel_available() is False
    monkeypatch.delenv("FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK")
    monkeypatch.setattr(re_mod, "_load_vllm_custom_ops", lambda: None)
    assert re_mod._fused_rope_kernel_available() is False


def test_get_rope_prefers_local_class_when_fused_op_available(monkeypatch):
    """With the custom op importable, get_rope must NOT hand out vLLM's object (whose ROCm dispatch is the torch path)."""
    monkeypatch.setattr(re_mod, "_load_vllm_custom_ops", lambda: object())
    monkeypatch.setattr(re_mod, "_load_vllm_get_rope", lambda: (_ for _ in ()).throw(AssertionError("vLLM factory must not be used")))
    monkeypatch.setattr(re_mod, "_should_prefer_torch_rope_fallback", lambda: False)
    built = {}

    class _Local(_FrontierLike):
        def __init__(self, *a, **k):
            torch.nn.Module.__init__(self); built["args"] = a
    monkeypatch.setattr(re_mod, "RotaryEmbedding", _Local)
    monkeypatch.setattr(re_mod, "_LOCAL_ROPE_DICT", {})
    rope = re_mod.get_rope(HEAD_SIZE, rotary_dim=HEAD_SIZE, max_position=MAX_POS, base=BASE, is_neox_style=True, rope_scaling=None)
    assert isinstance(rope, _Local) and built["args"][0] == HEAD_SIZE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU with the bundled vLLM")
def test_vllm_kernel_matches_reference(monkeypatch):
    pytest.importorskip("vllm")
    monkeypatch.delenv("FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK", raising=False)
    monkeypatch.delenv("FRONTIER_PROFILING_FORCE_TORCH_FALLBACK", raising=False)
    torch.manual_seed(3)
    heads, n = 8, 256
    rope = re_mod.get_rope(HEAD_SIZE, rotary_dim=HEAD_SIZE, max_position=MAX_POS, base=BASE, is_neox_style=True,
                           rope_scaling=None, dtype=torch.bfloat16)
    assert re_mod.rope_impl_name(rope) == "vllm_kernel", type(rope)
    assert isinstance(rope, re_mod.RotaryEmbedding)  # our class -> vllm._custom_ops.rotary_embedding, one fused kernel
    positions = torch.randint(0, MAX_POS, (n,), device="cuda")
    q = torch.randn(n, heads * HEAD_SIZE, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n, 2 * HEAD_SIZE, device="cuda", dtype=torch.bfloat16)
    q_in, k_in = q.clone(), k.clone()
    q_out, k_out = rope(positions, q, k)
    q_ref = reference_rope(q_in.float().cpu(), positions.cpu(), heads, is_neox_style=True)
    k_ref = reference_rope(k_in.float().cpu(), positions.cpu(), 2, is_neox_style=True)
    # bf16 cache and inputs: |x| up to ~4 -> two products of 2^-8-relative operands ~ 3e-2 (measured 0.017-0.036 for every
    # correct implementation on this image: fused op, SGLang's kernel, vLLM native); the old bug gave 6-9
    assert (q_out.float().cpu().double() - q_ref).abs().max().item() <= 4e-2
    assert (k_out.float().cpu().double() - k_ref).abs().max().item() <= 4e-2
