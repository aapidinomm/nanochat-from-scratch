import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nanochat_from_scratch.config import GPTConfig
from nanochat_from_scratch.model import RMSNorm
from nanochat_from_scratch.block import TransformerBlock

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        assert config.vocab_size > 0
        assert config.max_seq_len > 0
        assert config.num_layers > 0
        assert config.hidden_size > 0
        assert config.num_heads > 0
        assert config.num_kv_heads > 0

        assert config.hidden_size % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0
        assert (config.hidden_size // config.num_heads) % 2 == 0

        # Convert token ids into hidden vectors
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        # Stack N Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size = config.hidden_size, 
                num_heads = config.num_heads, 
                num_kv_heads = config.num_kv_heads
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm()

        # Map hidden vectors to vocabulary logits
        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias = False
        )

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert 0 < T <= self.config.max_seq_len

        # Convert tokens to embedding vectors
        x = self.token_embedding(input_ids)

        # Input passes through all the transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            assert targets.shape == input_ids.shape

            loss = F.cross_entropy(
                rearrange(logits, "b t v -> (b t) v").float(),
                rearrange(targets, "b t -> (b t)")
            )
        return logits, loss

        
        
        
        
        
        