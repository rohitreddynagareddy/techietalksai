# common.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tiny_transformer import TinyTransformer

# ——— Vocabulary & Device ———

def load_vocab(path="data/tiny_corpus.txt"):
    text = open(path, 'r').read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return text, chars, stoi, itos

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ——— Model Factory ———

# def build_model(vocab_size,
#                 d_model=256,
#                 nhead=8,
#                 num_layers=4,
#                 dim_feedforward=1024,
#                 dropout=0.2):
#     """
#     Bigger model: more layers, heads, and hidden dims for richer representations.
#     """
#     return TinyTransformer(
#         vocab_size=vocab_size,
#         d_model=d_model,
#         nhead=nhead,
#         num_layers=num_layers,
#         dim_feedforward=dim_feedforward,
#         dropout=dropout
#     )

from tiny_transformer import PositionalEncoding, TinyTransformer

def build_model(
    vocab_size,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.3,
):
    """
    Standard Transformer with sinusoidal positional encodings.
    """
    return TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pos_enc=PositionalEncoding(d_model)   # pass in your existing PosEnc
    )


# ——— Training Hyperparameters ———

# Sequence length & batching
SEQ_LEN     = 64       # longer context windows
BATCH_SIZE  = 32       # larger batch

# Training schedule
EPOCHS      = 500      # fewer epochs with more capacity
LR          = 5e-4     # AdamW-friendly LR
LABEL_SMOOTH = 0.1     # prevent overconfidence
CLIP_NORM   = 1.0      # gradient clipping

# Optimizer & scheduler builders

def get_optimizer(model):
    """
    AdamW is often better with Transformers.
    """
    return AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

def get_scheduler(optimizer):
    """
    Cosine decay from initial LR down to near zero over all epochs.
    """
    return CosineAnnealingLR(optimizer, T_max=EPOCHS)
