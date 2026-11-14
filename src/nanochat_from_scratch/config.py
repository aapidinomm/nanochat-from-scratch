from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 512
    max_seq_len: int = 128
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int = 2

    
    