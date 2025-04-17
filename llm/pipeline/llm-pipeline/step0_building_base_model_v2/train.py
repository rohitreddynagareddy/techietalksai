# train.py

import time
import torch
import torch.nn as nn
from tqdm import tqdm

from common import (
    load_vocab,
    get_device,
    build_model,
    SEQ_LEN,
    BATCH_SIZE,
    EPOCHS,
    LABEL_SMOOTH,
    CLIP_NORM,
    get_optimizer,
    get_scheduler,
)

# 1. Load vocabulary & data
text, chars, stoi, itos = load_vocab("data/tiny_corpus.txt")
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

# 2. Device & model
device = get_device()
model = build_model(vocab_size=len(chars)).to(device)

# 3. Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer)

# 4. Batch generator
def get_batch():
    idx = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    inp = torch.stack([data[i : i + SEQ_LEN] for i in idx])
    tgt = torch.stack([data[i + 1 : i + 1 + SEQ_LEN] for i in idx])
    return inp.t().to(device), tgt.t().to(device)  # (seq_len, batch)

# --- Training ---
total_start = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training started on {device}")

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    total_loss = 0.0

    for _ in range(100):  # 100 batches per epoch
        inp, tgt = get_batch()
        logits = model(inp)  # (seq_len, batch, vocab)
        loss = criterion(
            logits.reshape(-1, len(chars)),
            tgt.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()

    # step LR scheduler
    scheduler.step()

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / 100
    print(f"[Epoch {epoch:4d}/{EPOCHS}] Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

total_time = time.time() - total_start
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training finished in {total_time/60:.2f}m ({total_time:.1f}s)")

# 5. Save model
torch.save(model.state_dict(), "tiny_transformer.pth")
print("Model parameters saved to tiny_transformer.pth")
