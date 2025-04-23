# import torch
# import torch.nn as nn

# print("\n🔹 Step 3: Tiny Transformer 🔹\n")

# # Vocabulary size
# vocab_size = 8  # Assume small vocabulary from previous step
# embedding_dim = 16

# # Model: Embedding + Linear projection
# class TinyTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.fc = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, x):
#         print(f"📥 Input tokens: {x}")
#         x = self.embedding(x)
#         print(f"🧠 Embedded vectors: {x}")
#         x = self.fc(x)
#         print(f"📤 Output logits: {x}")
#         return x

# # Instantiate
# model = TinyTransformer()

# # Dummy input (batch of tokens)
# x = torch.randint(0, vocab_size, (4,))
# print(f"\n🚀 Random input tokens: {x}\n")

# # Forward pass
# out = model(x)
# print(f"\n🎯 Output shape: {out.shape}")

import torch
import torch.nn as nn

print("\n🔹 Step 3: Tiny Transformer 🔹\n")

# ----------------------------
# Vocabulary size and embedding size
# ----------------------------

# Assume small vocabulary from previous step ('h', 'e', 'l', 'o', etc.)
vocab_size = 8  # 8 unique tokens
embedding_dim = 16  # Each token becomes a 16-dimensional vector

# ----------------------------
# TinyTransformer Model
# ----------------------------
# The model has:
# 1. An Embedding layer: Maps token IDs to dense 16D vectors
# 2. A Linear layer: Projects 16D vectors back to vocab_size logits

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Lookup table: token index → 16D learned vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear transformation: 16D vector → vocab_size logits
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        print(f"📥 Input tokens (IDs):\n{x}\n")
        
        # Step 1: Embedding
        x = self.embedding(x)
        print(f"🧠 Embedded vectors (after nn.Embedding):\n{x}\n")
        
        # Step 2: Linear projection to logits
        x = self.fc(x)
        print(f"📤 Output logits (after nn.Linear):\n{x}\n")
        
        return x

# ----------------------------
# Instantiate the Model
# ----------------------------
model = TinyTransformer()

# ----------------------------
# Create Dummy Input (Batch of Tokens)
# ----------------------------
# Random 4 tokens (batch size = 4)
x = torch.randint(0, vocab_size, (4,))  # Shape: [4]
print(f"\n🚀 Randomly generated input tokens:\n{x}\n")

# ----------------------------
# Forward Pass through the Model
# ----------------------------
out = model(x)

# Output shape explanation
# Input Shape: [4] ➔ After Embedding: [4, 16] ➔ After Linear: [4, 8]
print(f"🎯 Final Output shape (batch_size, vocab_size): {out.shape}")

# ----------------------------
# Quick Concept Recap
# ----------------------------
print("""
🔵 Quick Concept:
- Tokens are discrete IDs → embedded into 16D continuous vectors.
- Each vector is linearly transformed into vocab_size scores (logits).
- These logits can be later used to predict the next token using softmax.

This is the foundation of GPT, BERT, and all Transformer models.
""")

