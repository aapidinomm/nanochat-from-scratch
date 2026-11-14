import torch

from nanochat_from_scratch.block import TransformerBlock


def test_block_shape_and_gradients():
    torch.manual_seed(42)

    block = TransformerBlock(
        hidden_size=64,
        num_heads=8,
        num_kv_heads=2,
    )

    x = torch.randn(
        2, 16, 64, requires_grad=True
    )

    y = block(x)

    assert y.shape == (2, 16, 64)

    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for param in block.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

def test_block_identity_when_branches_zero():
    block = TransformerBlock(64, 8, 2)

    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.mlp.proj.weight.zero_()

    x = torch.randn(2, 8, 64)
    y = block(x)

    torch.testing.assert_close(y, x)

def test_block_matches_reference():
    torch.manual_seed(42)

    block = TransformerBlock(64, 8, 2)
    x = torch.randn(2, 8, 64)

    actual = block(x)

    h = x + block.attn(block.attn_norm(x))
    expected = h + block.mlp(block.mlp_norm(h))

    torch.testing.assert_close(
        actual, expected,
        atol=1e-6, rtol=1e-5,
    )

def test_block_causality():
    torch.manual_seed(42)

    block = TransformerBlock(64, 8, 2)
    x = torch.randn(1, 8, 64)

    original = block(x)

    modified = x.clone()
    modified[:, 5:, :] += 100

    updated = block(modified)

    torch.testing.assert_close(
        original[:, :5, :],
        updated[:, :5, :],
        atol=1e-5,
        rtol=1e-5,
    )