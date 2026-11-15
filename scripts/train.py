
import math
import time
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict

import torch

from nanochat_from_scratch.config import GPTConfig
from nanochat_from_scratch.gpt import GPT
from nanochat_from_scratch.data import CharTokenizer, get_batch


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    def autocast_context():
        if use_bf16:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        return nullcontext()

    # Prepare training and validation data.
    text = Path("data/tiny_shakespeare.txt").read_text(
        encoding="utf-8"
    )

    tokenizer = CharTokenizer(text)

    all_ids = torch.tensor(
        tokenizer.encode(text), dtype=torch.long
    )

    split = int(0.9 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
    )

    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
    )

    batch_size = 32
    max_steps = 2000
    eval_interval = 100
    eval_batches = 20
    log_interval = 50

    print("Device:", device)
    print("BF16 autocast:", use_bf16)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))
    print("Random baseline:", math.log(config.vocab_size))

    def next_batch(split_name):
        data = train_ids if split_name == "train" else val_ids

        return get_batch(
            data,
            batch_size=batch_size,
            seq_len=config.max_seq_len,
            device=device,
        )

    @torch.no_grad()
    def evaluate():
        model.eval()
        results = {}

        for split_name in ("train", "val"):
            losses = []

            for _ in range(eval_batches):
                x, y = next_batch(split_name)

                with autocast_context():
                    _, loss = model(x, y)

                losses.append(loss.item())

            results[split_name] = sum(losses) / len(losses)

        model.train()
        return results

    def sync_gpu():
        if device.type == "cuda":
            torch.cuda.synchronize()

    Path("checkpoints").mkdir(exist_ok=True)

    best_val_loss = float("inf")
    model.train()

    sync_gpu()
    start = time.perf_counter()

    for step in range(1, max_steps + 1):
        x, y = next_batch("train")

        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        if step % log_interval == 0:
            sync_gpu()
            elapsed = time.perf_counter() - start

            tokens_per_sec = (
                batch_size * config.max_seq_len
                * log_interval / elapsed
            )

            print(
                f"step={step:04d} "
                f"train_batch_loss={loss.item():.4f} "
                f"grad_norm={grad_norm.item():.3f} "
                f"tokens/sec={tokens_per_sec:,.0f}"
            )

            start = time.perf_counter()

        if step % eval_interval == 0:
            metrics = evaluate()

            train_loss = metrics["train"]
            val_loss = metrics["val"]

            print(
                f"EVAL step={step:04d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    {
                        "step": step,
                        "config": asdict(config),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "chars": tokenizer.chars,
                        "val_loss": val_loss,
                    },
                    "checkpoints/best.pt",
                )

    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
