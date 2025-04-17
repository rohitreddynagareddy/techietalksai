# generate.py

import torch
import torch.nn.functional as F

from common import load_vocab, get_device, build_model

# 1. Load vocab & device
text, chars, stoi, itos = load_vocab("data/tiny_corpus.txt")
device = get_device()

# 2. Rebuild model & load weights
model = build_model(vocab_size=len(chars)).to(device)
checkpoint = torch.load("tiny_transformer.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()


def filter_logits(logits, top_k=0, top_p=0.0):
    """
    Apply Top-K and/or Top-P (nucleus) filtering on logits.
    If top_k=0 or >= vocab_size, skips Top-K.
    If top_p=0.0 or >=1.0, skips Top-P.
    """
    vocab_size = logits.size(-1)

    # Top-K
    if 0 < top_k < vocab_size:
        k = min(top_k, vocab_size)
        values, _ = torch.topk(logits, k)
        min_val = values[-1]
        logits = torch.where(
            logits < min_val,
            torch.full_like(logits, -1e10),
            logits
        )

    # Top-P (nucleus)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # identify cutoff index
        cutoff_mask = cum_probs > top_p
        if cutoff_mask.any():
            cutoff_idx = cutoff_mask.nonzero()[0, 0].item()
            threshold = sorted_logits[cutoff_idx]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, -1e10),
                logits
            )

    return logits


@torch.no_grad()
def generate(
    start_str: str = "To be",
    length: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """
    Autoregressive character-level generation with temperature + top-k/top-p filtering.
    """
    # encode prompt
    idxs = [stoi.get(ch, 0) for ch in start_str]  # fallback to 0 if unseen
    input_ids = torch.tensor(idxs, dtype=torch.long, device=device).unsqueeze(1)  # (seq, 1)

    for _ in range(length):
        logits = model(input_ids)[-1, 0] / temperature  # (vocab,)
        filtered = filter_logits(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=0)

    return "".join(itos[int(i)] for i in input_ids.squeeze().tolist())


if __name__ == "__main__":
    prompt = "Machine Learning allows"
    print(f"\n--- Generation (prompt: \"{prompt}\") ---\n")
    sample = generate(prompt, length=100, temperature=0.7, top_k=50, top_p=0.8)
    print(sample)
    
    prompt = "Machine Learning"
    print(f"\n--- Generation (prompt: \"{prompt}\") ---\n")
    sample = generate(prompt, length=100, temperature=0.7, top_k=50, top_p=0.8)
    print(sample)

    prompt = "Machine"
    print(f"\n--- Generation (prompt: \"{prompt}\") ---\n")
    sample = generate(prompt, length=100, temperature=0.7, top_k=50, top_p=0.8)
    print(sample)
