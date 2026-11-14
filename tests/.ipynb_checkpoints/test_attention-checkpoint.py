
import torch
import math
import pytest
from einops import einsum, repeat
import torch.nn.functional as F
from einops import rearrange

from nanochat_from_scratch.attention import CausalSelfAttention
from nanochat_from_scratch.rope import (
    rotary_cos_sin,
    apply_rope,
)


def test_attention_shape_and_gradients():
    x = torch.randn(2, 4, 8, requires_grad=True)
    attention = CausalSelfAttention(
        hidden_size=8, num_heads=2
    )

    y = attention(x)

    assert y.shape == (2, 4, 8)

    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_causal_mask():
    torch.manual_seed(42)

    attention = CausalSelfAttention(8, 2)
    x = torch.randn(1, 5, 8)

    original = attention(x)

    modified = x.clone()
    modified[:, 3:, :] += 100

    updated = attention(modified)

    # Changing future tokens must not change
    # outputs for earlier token positions.
    torch.testing.assert_close(
        original[:, :3, :],
        updated[:, :3, :],
        atol=1e-6,
        rtol=1e-5,
    )


def test_matches_pytorch_sdpa():
    torch.manual_seed(42)

    attention = CausalSelfAttention(8, 2)
    x = torch.randn(2, 4, 8)

    actual = attention(x)

    q = rearrange(
        attention.q_proj(x),
        "b t (h d) -> b h t d",
        h=2,
    )
    k = rearrange(
        attention.k_proj(x),
        "b t (h d) -> b h t d",
        h=2,
    )
    v = rearrange(
        attention.v_proj(x),
        "b t (h d) -> b h t d",
        h=2,
    )


    cos, sin = rotary_cos_sin(
        seq_len=x.shape[1],
        head_dim=attention.head_dim,
        device=x.device,
    )

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    # Match the model's qk-norm before computing scores
    q = attention.q_norm(q)
    k = attention.k_norm(k)

    context = F.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )
    context = rearrange(
        context, "b h t d -> b t (h d)"
    )

    expected = attention.out_proj(context)

    torch.testing.assert_close(
        actual, expected, atol=1e-5, rtol=1e-4
    )

def test_qk_normalization():
    torch.manual_seed(42)

    attn = CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
    )

    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)

    q_norm = attn.q_norm(q)
    k_norm = attn.k_norm(k)

    for tensor in [q_norm, k_norm]:
        rms = tensor.square().mean(dim=-1).sqrt()

        torch.testing.assert_close(
            rms,
            torch.ones_like(rms),
            atol=1e-5,
            rtol=1e-5,
        )

def test_gqa_projection_dimensions():
    attention = CausalSelfAttention(
        hidden_size=512,
        num_heads=8,
        num_kv_heads=2,
    )

    assert attention.head_dim == 64
    assert attention.group_size == 4

    assert attention.q_proj.out_features == 512
    assert attention.k_proj.out_features == 128
    assert attention.v_proj.out_features == 128




@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_gqa_output_and_gradients(num_kv_heads):
    torch.manual_seed(42)

    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=8,
        num_kv_heads=num_kv_heads,
    )

    x = torch.randn(2, 5, 64, requires_grad=True)

    output = attention(x)

    assert output.shape == (2, 5, 64)

    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_gqa_causal_mask():
    torch.manual_seed(42)

    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=8,
        num_kv_heads=2,
    )

    x = torch.randn(1, 6, 64)

    original = attention(x)

    modified = x.clone()
    modified[:, 4:, :] += 100

    updated = attention(modified)

    torch.testing.assert_close(
        original[:, :4, :],
        updated[:, :4, :],
        atol=1e-5,
        rtol=1e-5,
    )





def manual_gqa(q, k, v):
    group_size = q.shape[1] // k.shape[1]

    k = repeat(
        k, "b g t d -> b (g r) t d", r=group_size
    )
    v = repeat(
        v, "b g t d -> b (g r) t d", r=group_size
    )

    scores = einsum(
        q, k, "b h t d, b h s d -> b h t s"
    ) / math.sqrt(q.shape[-1])

    T = q.shape[-2]
    mask = torch.ones(
        T, T, device=q.device, dtype=torch.bool
    ).tril()

    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)

    return einsum(
        weights, v, "b h t s, b h s d -> b h t d"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Native GQA test requires CUDA",
)
@pytest.mark.parametrize("kv_heads", [1, 2, 8])
def test_native_gqa_matches_manual(kv_heads):
    torch.manual_seed(42)

    q = torch.randn(2, 8, 7, 16, device="cuda", requires_grad=True)
    k = torch.randn(2, kv_heads, 7, 16, device="cuda", requires_grad=True)
    v = torch.randn(2, kv_heads, 7, 16, device="cuda", requires_grad=True)

    manual = manual_gqa(q, k, v)

    native = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=True,
        enable_gqa=(kv_heads != 8),
    )

    torch.testing.assert_close(
        native, manual, rtol=1e-4, atol=1e-4
    )

    manual_grads = torch.autograd.grad(
        manual.square().sum(), (q, k, v)
    )
    native_grads = torch.autograd.grad(
        native.square().sum(), (q, k, v)
    )

    for actual, expected in zip(native_grads, manual_grads):
        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-4
        )
