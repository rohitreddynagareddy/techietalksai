import torch
import torch.nn as nn
import math

print("\n🔹 Building a Mini Transformer Decoder Block 🔹\n")

# ----------------------------
# Hyperparameters
# ----------------------------
vocab_size = 8
embedding_dim = 16
block_size = 8  # context size (max tokens)
n_heads = 2
hidden_dim = 64  # size of feed-forward hidden layer

# ----------------------------
# Step 1: Helper Function - Create Causal Mask
# ----------------------------
def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask

# ----------------------------
# Step 2: Define the Mini Decoder Block
# ----------------------------
class MiniDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Embedding Layer (token embeddings)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Positional Embedding Layer
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        # 3. Multi-Head Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)

        # 4. Feedforward Network (MLP)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # 5. Layer Normalization Layers
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        # ----------------------------
        # Token Embedding + Position Embedding
        # ----------------------------
        token_embeds = self.token_embedding(x)  # (batch_size, seq_len, embedding_dim)
        position_ids = torch.arange(seq_len, device=device)
        position_embeds = self.position_embedding(position_ids)

        x = token_embeds + position_embeds  # Add token + position embeddings
        print(f"\n🔹 After Embedding: {x.shape}")

        # ----------------------------
        # Causal Masking
        # ----------------------------
        causal_mask = create_causal_mask(seq_len).to(device)

        # ----------------------------
        # Multi-Head Self Attention with Mask
        # ----------------------------
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=causal_mask)
        print(f"🔹 After Self-Attention: {attn_output.shape}")

        # ----------------------------
        # First Add & Norm
        # ----------------------------
        x = self.ln1(x + attn_output)

        # ----------------------------
        # Feedforward Network
        # ----------------------------
        ff_output = self.feedforward(x)
        print(f"🔹 After Feedforward: {ff_output.shape}")

        # ----------------------------
        # Second Add & Norm
        # ----------------------------
        x = self.ln2(x + ff_output)

        return x

# ----------------------------
# Step 3: Test the Mini Decoder Block
# ----------------------------

# Instantiate model
model = MiniDecoderBlock()

# Dummy input: batch of 2 sequences, each of length 5
dummy_input = torch.randint(0, vocab_size, (2, 5))  # (batch_size=2, seq_len=5)

# Forward pass
output = model(dummy_input)

print(f"\n✅ Final Output Shape: {output.shape}")
