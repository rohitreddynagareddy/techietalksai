# tiny_transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory=None):
        # tgt: (seq_len, batch)
        x = self.token_emb(tgt)                           # → (seq_len, batch, d_model)
        x = self.pos_enc(x)
        # since no encoder, memory can be None → use self as a decoder-only
        out = self.transformer(x, x)
        return self.fc_out(out)                           # → (seq_len, batch, vocab_size)

