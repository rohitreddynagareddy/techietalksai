# import torch
# import torch.nn as nn

# print("\n🔹 Step 4: Forward Pass with Debugging 🔹\n")

# # Step 1: Hyperparameters
# vocab = sorted(set("hello world"))
# vocab_size = len(vocab)
# embedding_dim = 16

# print(f"🔠 Vocab: {vocab}")
# print(f"🔢 Vocab Size: {vocab_size}")
# print(f"📏 Embedding Dimension: {embedding_dim}")

# # Step 2: Mappings
# char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
# idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# print(f"🗺️ char_to_idx: {char_to_idx}")
# print(f"🗺️ idx_to_char: {idx_to_char}")

# # Step 3: Encode corpus
# corpus = "hello"
# encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)
# print(f"\n🧩 Encoded input sequence: {encoded}")

# # Step 4: Define Model
# class TinyTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.fc = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, x):
#         print(f"\n📥 [Forward] Input tokens: {x}")
#         x_embed = self.embedding(x)
#         print(f"🔍 [Forward] After Embedding: {x_embed.shape}\n{x_embed}")

#         x_out = self.fc(x_embed)
#         print(f"🎯 [Forward] After Linear: {x_out.shape}\n{x_out}")

#         return x_out

# # Step 5: Run model
# model = TinyTransformer()

# print("\n🚀 Running Forward Pass...\n")
# logits = model(encoded)

# print("\n✅ Forward Pass Done")
# print(f"📝 Output Logits shape: {logits.shape} [Should be: (input_length, vocab_size)]")


import torch
import torch.nn as nn

print("\n🔹 Step 4: Forward Pass with Debugging 🔹\n")

# ----------------------------
# Step 1: Hyperparameters
# ----------------------------
# Create a vocabulary from the string "hello world"
vocab = sorted(set("hello world"))
vocab_size = len(vocab)  # Unique characters
embedding_dim = 16       # Each character maps to a 16D vector

print(f"🔠 Vocab (Sorted Unique Characters): {vocab}")
print(f"🔢 Vocab Size: {vocab_size}")
print(f"📏 Embedding Dimension: {embedding_dim}")

# ----------------------------
# Step 2: Character-Index Mappings
# ----------------------------
# Create lookup tables for encoding and decoding characters
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

print(f"🗺️ Character to Index Mapping:\n{char_to_idx}")
print(f"🗺️ Index to Character Mapping:\n{idx_to_char}")

# ----------------------------
# Step 3: Encode Input Corpus
# ----------------------------
# Encode a small text ("hello") into token indices
corpus = "hello"
encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)

print(f"\n🧩 Encoded Input Sequence (hello → IDs):\n{encoded}")
print(f"📏 Shape of Encoded Input: {encoded.shape}")

# ----------------------------
# Step 4: Define the Tiny Transformer Model
# ----------------------------
# It has:
# 1. An Embedding Layer: token ID → 16D vector
# 2. A Linear Layer: 16D vector → vocab_size logits

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Token Embedding: maps each token to a 16D vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Linear Projection: maps each 16D vector back to vocab_size logits
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        print(f"\n📥 [Forward] Input token IDs:\n{x}\n")
        
        # Step 1: Token Embedding
        x_embed = self.embedding(x)
        print(f"🔍 [Forward] After Embedding Layer:")
        print(f"Shape: {x_embed.shape} [Should be: (input_length, embedding_dim)]")
        print(x_embed)

        # Step 2: Linear Layer
        x_out = self.fc(x_embed)
        print(f"\n🎯 [Forward] After Linear Projection Layer:")
        print(f"Shape: {x_out.shape} [Should be: (input_length, vocab_size)]")
        print(x_out)

        return x_out

# ----------------------------
# Step 5: Instantiate and Run the Model
# ----------------------------
model = TinyTransformer()

print("\n🚀 Running Forward Pass Through the Model...\n")

# Pass the encoded input through the model
logits = model(encoded)

print("\n✅ Forward Pass Completed Successfully!")
print(f"📝 Final Output Logits Shape: {logits.shape}")
print(f"🔎 Shape Meaning: (input_length = {encoded.shape[0]}, vocab_size = {vocab_size})")


