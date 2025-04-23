import torch
import torch.nn as nn
import torch.optim as optim

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dummy Vocabulary and Example Data (like 'hello world')
vocab = [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
vocab_size = len(vocab)
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# Training Text Example
train_text = "hello world"
encoded = torch.tensor([char_to_idx[c] for c in train_text if c in char_to_idx], dtype=torch.long)

# Hyperparameters
embedding_dim = 32
block_size = 8
learning_rate = 0.001
epochs = 500

# 🛠️ Mini-GPT Model Definition
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 🛠️ FIXED: Make position embedding larger to avoid overflow
        self.position_embedding = nn.Embedding(2 * block_size, embedding_dim)
        
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Token + Position Embedding
        token_emb = self.token_embedding(x)
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=x.device)
        position_emb = self.position_embedding(position_ids)
        
        # Add embeddings
        x = token_emb + position_emb

        # Self-Attention with Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.layernorm1(x + attn_out)

        # Feedforward
        ff_out = self.feedforward(x)
        x = self.layernorm2(x + ff_out)

        logits = self.output_layer(x)
        return logits

# Initialize model
model = MiniGPT(vocab_size, embedding_dim, block_size).to(device)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training Loop
print("\n🚀 Training Mini-GPT...\n")
for epoch in range(1, epochs + 1):
    model.train()
    
    # Take random block
    start_idx = torch.randint(0, len(encoded) - block_size, (1,)).item()
    x_train = encoded[start_idx:start_idx + block_size].unsqueeze(0).to(device)
    y_train = encoded[start_idx + 1:start_idx + block_size + 1].unsqueeze(0).to(device)

    optimizer.zero_grad()
    outputs = model(x_train)

    loss = criterion(outputs.view(-1, vocab_size), y_train.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

print("\n✅ Training Complete!")

# Save model weights
torch.save(model.state_dict(), "mini_gpt.pth")

print("\n💾 Model saved as mini_gpt.pth")

