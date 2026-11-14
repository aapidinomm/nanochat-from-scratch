
import torch
import torch.nn.functional as F

from nanochat_from_scratch.config import GPTConfig
from nanochat_from_scratch.gpt import GPT


def make_small_model():
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,
    )
    return GPT(config)


def test_gpt_logits_shape():
    model = make_small_model()
    tokens = torch.randint(0, 100, (2, 16))

    logits, loss = model(tokens)

    assert logits.shape == (2, 16, 100)
    assert loss is None


def test_gpt_loss_and_gradients():
    model = make_small_model()
    tokens = torch.randint(0, 100, (2, 17))

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 16, 100)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_gpt_cross_entropy_reference():
    model = make_small_model()
    tokens = torch.randint(0, 100, (2, 9))

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(inputs, targets)

    expected = F.cross_entropy(
        logits.reshape(-1, 100).float(),
        targets.reshape(-1),
    )

    torch.testing.assert_close(loss, expected)


def test_gpt_causality():
    torch.manual_seed(42)

    model = make_small_model()
    tokens = torch.randint(0, 100, (1, 12))

    original, _ = model(tokens)

    modified = tokens.clone()
    modified[:, 8:] = torch.randint(0, 100, (1, 4))

    updated, _ = model(modified)

    torch.testing.assert_close(
        original[:, :8, :],
        updated[:, :8, :],
        atol=1e-5,
        rtol=1e-5,
    )
