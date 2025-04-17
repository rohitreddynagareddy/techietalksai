import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]

class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pos_enc: nn.Module = None
    ):
        super().__init__()
        # token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # positional encoding: either provided or default sinusoidal
        self.pos_enc = pos_enc if pos_enc is not None else PositionalEncoding(d_model)

        # build transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: (seq_len, batch)
        x = self.token_emb(tgt)  # (seq_len, batch, d_model)
        x = self.pos_enc(x)      # add positional encoding
        # decoder: using tgt as both query and memory for decoder-only
        out = self.transformer(
            x,
            x,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(out)  # (seq_len, batch, vocab_size)
