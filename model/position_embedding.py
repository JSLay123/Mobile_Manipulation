import math

import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=910):
        super(PositionalEmbedding, self).__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        if x.size(1) > self.pos_enc.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_seq_len {self.pos_enc.size(1)}")
        x = x + self.pos_enc[:, :x.size(1), :]
        return x