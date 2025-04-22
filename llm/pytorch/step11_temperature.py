import torch
import torch.nn as nn
import torch.optim as optim

print("\n🔹 Step 6++: Inference with Top-k + Temperature Sampling + Long Context 🔹\n")

# ----------------------------
# Step 1: Hyperparameters
# ----------------------------
vocab = sorted(set("hello world"))
vocab_size = len(vocab)
embedding_dim = 16
learning_rate = 0.01
epochs = 300

block_size = 8  # Max context size (for generation sliding window)

print(f"🔠 Vocab: {vocab}")
print(f"🔢 Vocab Size: {vocab_size}")
print(f"📏 Embedding Dimension: {embedding_dim}")
print(f"📏 Block Size: {block_size}")
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
# Step 7: Top-k + Temperature Sampling Function
# ----------------------------
def top_k_temperature_sampling(logits, k=3, temperature=1.0):
    """Apply top-k sampling with temperature control"""
    
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Top-k filtering
    topk_logits, topk_indices = torch.topk(scaled_logits, k)

    # Softmax over top-k logits
    probs = torch.softmax(topk_logits, dim=-1)

    # Sample from the distribution
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return topk_indices[sampled_idx]

# ----------------------------
# Step 8: Inference Function with Top-k + Temperature + Sliding Window
# ----------------------------
def generate_text(start_char, max_length=50, top_k_value=3, temperature=1.0):
    print(f"\n🧠 Starting generation from '{start_char}' (Top-k={top_k_value}, Temperature={temperature})...\n")
    model.eval()

    idx_input = torch.tensor([char_to_idx[start_char]], dtype=torch.long)
    generated = [start_char]

    for _ in range(max_length):
        # Sliding window context
        if idx_input.shape[0] > block_size:
            idx_input = idx_input[-block_size:]

        output_logits = model(idx_input)

        last_logits = output_logits[-1]

        next_token_idx = top_k_temperature_sampling(last_logits, k=top_k_value, temperature=temperature).item()

        next_token_char = idx_to_char[next_token_idx]
        generated.append(next_token_char)

        idx_input = torch.cat([idx_input, torch.tensor([next_token_idx])])

    final_text = ''.join(generated)
    print(f"📝 Generated Text:\n{final_text}\n")
    return final_text

# ----------------------------
# Step 9: Test Different Temperatures
# ----------------------------
generate_text('h', max_length=50, top_k_value=3, temperature=1.0)  # Normal randomness
generate_text('h', max_length=50, top_k_value=3, temperature=0.7)  # More focused
generate_text('h', max_length=50, top_k_value=3, temperature=1.5)  # Very creative/random

# ----------------------------
# Quick Concept Recap
# ----------------------------
print("""
🔵 Quick Concept Recap:
- **Top-k Sampling**: Only sample among top-k probable tokens.
- **Temperature**:
    - >1.0 → Soften logits → More random choices
    - <1.0 → Sharpen logits → More confident choices
- **Sliding Context Window**: Only last N tokens are fed during inference.

Together, these make your generation more realistic and controllable like real GPT models!
""")
