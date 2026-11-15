
import math
import torch

from nanochat_from_scratch.config import GPTConfig
from nanochat_from_scratch.gpt import GPT


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = GPTConfig(
        vocab_size=64,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
    )

    model = GPT(config).to(device)
    model.train()

    num_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")

    # Create a fixed training batch.
    # Do NOT regenerate these tokens each step.
    tokens = torch.randint(
        0, config.vocab_size,
        (2, 17),
        device=device,
    )

    input_ids = tokens[:, :-1]  # [2, 16]
    targets = tokens[:, 1:]     # [2, 16]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=0.0,
    )

    print(f"Random baseline loss: {math.log(config.vocab_size):.4f}")

    for step in range(401):
        optimizer.zero_grad(set_to_none=True)

        logits, loss = model(input_ids, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        if step % 25 == 0:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                accuracy = (
                    predictions == targets
                ).float().mean().item()

            print(
                f"step={step:03d} | "
                f"loss={loss.item():.4f} | "
                f"accuracy={accuracy:.2%} | "
                f"grad_norm={grad_norm.item():.3f}"
            )

    model.eval()
    with torch.no_grad():
        logits, loss = model(input_ids, targets)
        accuracy = (
            logits.argmax(dim=-1) == targets
        ).float().mean().item()

    print(f"Final loss: {loss.item():.4f}")
    print(f"Final accuracy: {accuracy:.2%}")

    
    # Autoregressive generation from the first token.
    model.eval()

    generated = tokens[:1, :1].clone()

    with torch.no_grad():
        for _ in range(tokens.shape[1] - 1):
            context = generated[:, -config.max_seq_len:]

            logits, _ = model(context)

            # Greedy decoding: choose the most likely next token.
            next_token = logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

            generated = torch.cat(
                [generated, next_token], dim=1
            )

    print("Original: ", tokens[0].tolist())
    print("Generated:", generated[0].tolist())
    print("Exact match:", torch.equal(generated, tokens[:1]))



if __name__ == "__main__":
    main()
