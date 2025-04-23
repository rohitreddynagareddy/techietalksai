import torch
import torch.nn as nn
import torch.optim as optim

print("\n🔹 Step 6+: Inference with Top-k Sampling and Long Context 🔹\n")

# ----------------------------
# Step 1: Hyperparameters
# ----------------------------
vocab = sorted(set("hello world"))
vocab_size = len(vocab)
embedding_dim = 16
learning_rate = 0.01
epochs = 300

block_size = 8  # Max context length to consider (like GPTs)

print(f"🔠 Vocab: {vocab}")
print(f"🔢 Vocab Size: {vocab_size}")
print(f"📏 Embedding Dimension: {embedding_dim}")
print(f"📏 Block Size (Context Window): {block_size}")
print(f"⚡ Learning Rate: {learning_rate}")
print(f"🔁 Epochs: {epochs}")

# ----------------------------
# Step 2: Character-Index Mappings
# ----------------------------
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# ----------------------------
# Step 3: Encode Corpus
# ----------------------------
corpus = "hello"
encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)

x = encoded[:-1]
y = encoded[1:]

# ----------------------------
# Step 4: Model
# ----------------------------
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x_embed = self.embedding(x)
        x_out = self.fc(x_embed)
        return x_out

model = TinyTransformer()

# ----------------------------
# Step 5: Loss and Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Step 6: Training Loop
# ----------------------------
print("\n🚀 Training Tiny Transformer...\n")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        print(f"📅 Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

print("\n✅ Training Completed Successfully!")

# ----------------------------
# Step 7: Top-k Sampling Function
# ----------------------------
def top_k_sampling(logits, k=3):
    """Sample from top-k highest probability logits"""
    topk_logits, topk_indices = torch.topk(logits, k)
    probs = torch.softmax(topk_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return topk_indices[sampled_idx]

# ----------------------------
# Step 8: Inference Function with Top-k + Long Context
# ----------------------------
def generate_text(start_char, max_length=50, top_k_value=3):
    print(f"\n🧠 Starting generation from '{start_char}'...\n")
    model.eval()

    idx_input = torch.tensor([char_to_idx[start_char]], dtype=torch.long)
    generated = [start_char]

    for _ in range(max_length):
        # Trim context to last block_size tokens
        if idx_input.shape[0] > block_size:
            idx_input = idx_input[-block_size:]

        output_logits = model(idx_input)

        # Take last token's logits
        last_logits = output_logits[-1]

        # Top-k sampling
        next_token_idx = top_k_sampling(last_logits, k=top_k_value).item()

        next_token_char = idx_to_char[next_token_idx]
        generated.append(next_token_char)

        # Append new token to context
        idx_input = torch.cat([idx_input, torch.tensor([next_token_idx])])

    final_text = ''.join(generated)
    print(f"📝 Generated Text:\n{final_text}\n")
    return final_text

# ----------------------------
# Step 9: Test Generation
# ----------------------------
generate_text('h', max_length=50, top_k_value=3)
generate_text('l', max_length=50, top_k_value=3)
generate_text('w', max_length=50, top_k_value=3)

# ----------------------------
# Quick Concept Recap
# ----------------------------
print("""
🔵 Quick Concept Recap:
- **Top-k Sampling**: Instead of greedy picking the max, randomly sample among top-k options.
- **Long Context**: Keep only the latest 'block_size' tokens as input during generation.
- This helps simulate how GPT models maintain manageable context windows while generating.
""")
