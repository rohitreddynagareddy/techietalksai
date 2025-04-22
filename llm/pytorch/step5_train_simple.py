# import torch
# import torch.nn as nn
# import torch.optim as optim

# print("\n🔹 Step 5: Tiny Training Loop with Debugging 🔹\n")

# # Step 1: Hyperparameters
# vocab = sorted(set("hello world"))
# vocab_size = len(vocab)
# embedding_dim = 16
# learning_rate = 0.01
# epochs = 100

# print(f"🔠 Vocab: {vocab}")
# print(f"🔢 Vocab Size: {vocab_size}")
# print(f"📏 Embedding Dimension: {embedding_dim}")
# print(f"⚡ Learning Rate: {learning_rate}")
# print(f"🔁 Epochs: {epochs}")

# # Step 2: Mappings
# char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
# idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# # Step 3: Encode Corpus
# corpus = "hello"
# encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)
# print(f"\n🧩 Encoded input: {encoded}")

# # Input: "h", "e", "l", "l"
# x = encoded[:-1]   # except last
# # Target: "e", "l", "l", "o"
# y = encoded[1:]    # shifted by 1

# print(f"\n📝 Inputs (x): {x}")
# print(f"🎯 Targets (y): {y}")

# # Step 4: Model Definition
# class TinyTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.fc = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, x):
#         x_embed = self.embedding(x)
#         x_out = self.fc(x_embed)
#         return x_out

# model = TinyTransformer()

# # Step 5: Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Step 6: Training Loop
# print("\n🚀 Starting Training...\n")
# for epoch in range(1, epochs + 1):
#     optimizer.zero_grad()

#     outputs = model(x)    # (batch_size, vocab_size)
#     loss = criterion(outputs, y)

#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0 or epoch == 1:
#         print(f"📅 Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

# print("\n✅ Training Completed!")

import torch
import torch.nn as nn
import torch.optim as optim

print("\n🔹 Step 5: Tiny Training Loop with Debugging 🔹\n")

# ----------------------------
# Step 1: Hyperparameters
# ----------------------------
vocab = sorted(set("hello world"))  # Unique sorted characters
vocab_size = len(vocab)
embedding_dim = 16
learning_rate = 0.01
epochs = 100

print(f"🔠 Vocab: {vocab}")
print(f"🔢 Vocab Size: {vocab_size}")
print(f"📏 Embedding Dimension: {embedding_dim}")
print(f"⚡ Learning Rate: {learning_rate}")
print(f"🔁 Epochs: {epochs}")

# ----------------------------
# Step 2: Character-to-Index Mappings
# ----------------------------
# Mapping from characters to IDs and back
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# ----------------------------
# Step 3: Encode Corpus (Input Preparation)
# ----------------------------
corpus = "hello"

# Encode the corpus: convert characters to integer IDs
encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)

print(f"\n🧩 Encoded Input Sequence (hello → IDs):\n{encoded}")

# Prepare Input (x) and Target (y) for prediction task
# Predict next character: "h"→"e", "e"→"l", etc.
x = encoded[:-1]  # Input: all except last
y = encoded[1:]   # Target: shifted by 1

print(f"\n📝 Inputs (x):\n{x}")
print(f"🎯 Targets (y):\n{y}")

# ----------------------------
# Step 4: Model Definition (Tiny Transformer)
# ----------------------------
# Simple model: Embedding Layer + Linear Output Layer

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Step 1: Token Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Step 2: Linear Projection to vocab_size
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x_embed = self.embedding(x)
        x_out = self.fc(x_embed)
        return x_out

# Instantiate the model
model = TinyTransformer()

# ----------------------------
# Step 5: Define Loss and Optimizer
# ----------------------------
# - CrossEntropyLoss: compares logits with ground-truth targets
# - Adam: adaptive optimizer that works well for small models

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Step 6: Training Loop
# ----------------------------
print("\n🚀 Starting Tiny Training...\n")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()  # Reset gradients

    outputs = model(x)    # Forward pass (outputs are logits)
    
    # Important: outputs shape [batch, vocab_size], targets shape [batch]
    loss = criterion(outputs, y)  # Compute loss

    loss.backward()  # Backpropagation (compute gradients)
    optimizer.step()  # Update model parameters

    # Debugging information
    if epoch % 10 == 0 or epoch == 1:
        print(f"📅 Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

print("\n✅ Tiny Training Completed!")

# ----------------------------
# Quick Concept Recap
# ----------------------------
print("""
🔵 Quick Concept Recap:
- We encode text into integer IDs.
- The model learns embeddings + a classifier.
- Training teaches the model to predict the next token.
- CrossEntropyLoss measures how far predictions are from true next tokens.
- Backpropagation updates weights to minimize loss.

This is the fundamental structure behind GPT, BERT, and LLM training!
""")
