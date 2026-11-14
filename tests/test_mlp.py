
import torch
import torch.nn.functional as F

from nanochat_from_scratch.mlp import MLP


def test_mlp_output_shape():
    mlp = MLP(hidden_size=512)
    x = torch.randn(2, 16, 512)

    output = mlp(x)

    assert output.shape == (2, 16, 512)


def test_mlp_gradients():
    mlp = MLP(hidden_size=64)
    x = torch.randn(2, 8, 64, requires_grad=True)

    output = mlp(x)
    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for param in mlp.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_mlp_token_independence():
    torch.manual_seed(42)

    mlp = MLP(hidden_size=64)
    x = torch.randn(2, 8, 64)

    original = mlp(x)

    modified = x.clone()
    modified[0, 3, :] += 100

    updated = mlp(modified)

    # Other tokens should remain unaffected.
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[0, 3] = False

    torch.testing.assert_close(
        original[mask],
        updated[mask],
        atol=1e-6,
        rtol=1e-5,
    )


def test_mlp_matches_reference():
    torch.manual_seed(42)

    mlp = MLP(hidden_size=64)
    x = torch.randn(2, 8, 64)

    actual = mlp(x)

    # Independently construct the expected ReLU² computation.
    hidden = F.linear(x, mlp.fc.weight)
    hidden = torch.relu(hidden).square()
    expected = F.linear(hidden, mlp.proj.weight)

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )
