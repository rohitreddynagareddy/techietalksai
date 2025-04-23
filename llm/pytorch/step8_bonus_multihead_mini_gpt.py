import torch
import torch.nn as nn
import torch.optim as optim
import math

print("\n🔹 Step 7: Build Mini-GPT 🔹\n")

# Step 1: Hyperparameters
vocab = sorted(set("hello world"))
vocab_size = len(vocab)
embedding_dim = 32
learning_rate = 0.001
epochs = 500
block_size = 8  # context size

print(f"🔠 Vocab: {vocab}")
print(f"🔢 Vocab Size: {vocab_size}")
print(f"📏 Embedding Dimension: {embedding_dim}")
print(f"⚡ Learning Rate: {learning_rate}")
print(f"📏 Context Block Size: {block_size}")

# Step 2: Mappings
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# Step 3: Encode Corpus
corpus = "hello world"
encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)

x = encoded[:-1]  # Inputs
y = encoded[1:]   # Targets

# Step 4: Model
# class MiniGPT(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.position_embedding = nn.Embedding(block_size, embedding_dim)

#         self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)
#         self.layer_norm1 = nn.LayerNorm(embedding_dim)

#         self.ff = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#         )
#         self.layer_norm2 = nn.LayerNorm(embedding_dim)

#         self.output_layer = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, x):
#         batch_size, seq_len = x.shape

#         token_emb = self.token_embedding(x)
#         position_ids = torch.arange(seq_len, device=x.device)
#         position_emb = self.position_embedding(position_ids)

#         x = token_emb + position_emb
#         print(f"\n🔍 After Embedding + Positional Encoding: {x.shape}")

#         attn_output, _ = self.self_attention(x, x, x, need_weights=False)
#         x = self.layer_norm1(x + attn_output)
#         print(f"🧠 After Self-Attention + Residual: {x.shape}")

#         ff_output = self.ff(x)
#         x = self.layer_norm2(x + ff_output)
#         print(f"⚡ After FeedForward + Residual: {x.shape}")

#         logits = self.output_layer(x)
#         print(f"🎯 Output logits shape: {logits.shape}")

#         return logits

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # ✅ FIX: Make position embedding large enough
        self.position_embedding = nn.Embedding(2 * block_size, embedding_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        token_emb = self.token_embedding(x)
        position_ids = torch.arange(seq_len, device=x.device)

        # ✅ This is now safe (position_ids will not exceed embedding size)
        position_emb = self.position_embedding(position_ids)

        x = token_emb + position_emb
        print(f"\n🔍 After Embedding + Positional Encoding: {x.shape}")

        attn_output, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.layer_norm1(x + attn_output)
        print(f"🧠 After Self-Attention + Residual: {x.shape}")

        ff_output = self.ff(x)
        x = self.layer_norm2(x + ff_output)
        print(f"⚡ After FeedForward + Residual: {x.shape}")

        logits = self.output_layer(x)
        print(f"🎯 Output logits shape: {logits.shape}")

        return logits


# Step 5: Instantiate Model
model = MiniGPT()

# Step 6: Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 7: Prepare Data
x_train = x.unsqueeze(0)  # Add batch dimension
y_train = y.unsqueeze(0)  # Add batch dimension

# Step 8: Training Loop
print("\n🚀 Training Mini-GPT...\n")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    outputs = model(x_train)

    outputs = outputs.view(-1, vocab_size)
    targets = y_train.view(-1)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        print(f"📅 Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

print("\n✅ Training Completed!")

# Step 9: Inference
def generate_text(start_char, max_length=20):
    print(f"\n🧠 Starting generation with '{start_char}'...\n")
    model.eval()

    idx_input = torch.tensor([[char_to_idx[start_char]]], dtype=torch.long)

    generated = [start_char]

    for _ in range(max_length):
        output_logits = model(idx_input)

        probs = torch.softmax(output_logits[:, -1, :], dim=-1)
        predicted_idx = torch.argmax(probs, dim=-1).item()
        predicted_char = idx_to_char[predicted_idx]

        generated.append(predicted_char)

        idx_input = torch.cat([idx_input, torch.tensor([[predicted_idx]])], dim=1)

        if idx_input.shape[1] > block_size:
            idx_input = idx_input[:, 1:]  # Slide window

    final_text = ''.join(generated)
    print(f"📝 Generated Text: {final_text}\n")
    return final_text

# Step 10: Test
generate_text('h', max_length=20)
generate_text('w', max_length=20)
