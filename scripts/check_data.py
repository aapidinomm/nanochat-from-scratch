from pathlib import Path
import torch

from nanochat_from_scratch.data import (
    CharTokenizer,
    get_batch,
)

text = Path("data/tiny_shakespeare.txt").read_text(
    encoding="utf-8"
)

tokenizer = CharTokenizer(text)
ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

split = int(0.9 * len(ids))
train_ids = ids[:split]
val_ids = ids[split:]

x, y = get_batch(
    train_ids,
    batch_size=4,
    seq_len=128,
    device=torch.device("cpu"),
)

print("Vocabulary size:", tokenizer.vocab_size)
print("Train tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))
print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("Input:", repr(tokenizer.decode(x[0].tolist())))
print("Target:", repr(tokenizer.decode(y[0].tolist())))