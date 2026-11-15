import torch

class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {
            ch: i for i, ch in enumerate(self.chars)
        }
        self.itos = {
            i: ch for i, ch in enumerate(self.chars)
        }
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.itos[i] for i in ids])


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
):
    assert data.ndim == 1
    assert batch_size > 0 and seq_len > 0
    assert len(data) > seq_len

    # Random sequence starting offsets.
    starts = torch.randint(0, len(data)-seq_len, (batch_size, ))
    offsets = torch.arange(seq_len)

    # [B, T], predicting one character ahead.
    # Generates a batch of input sequences by adding consecutive offsets to each randomly selected starting position and fetching the corresponding tokens from data.
    x = data[starts[:, None] + offsets]
    # Generates the corresponding target sequences by shifting each input position one token forward, enabling GPT to learn next-token prediction.
    y = data[starts[:, None] + offsets + 1]
    
    return x.to(device), y.to(device)





        

    
        

        

        