# import torch
# import torch.nn as nn
# import torch.optim as optim

# print("\n🔹 Step 6: Inference and Text Generation 🔹\n")

# # Step 1: Hyperparameters
# vocab = sorted(set("hello world"))
# vocab_size = len(vocab)
# embedding_dim = 16
# learning_rate = 0.01
# epochs = 300   # Training a bit more for better results

# # Step 2: Mappings
# char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
# idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# # Step 3: Encode Corpus
# corpus = "hello"
# encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)

# x = encoded[:-1]
# y = encoded[1:]

# # Step 4: Model
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
# print("\n🚀 Training Model...\n")
# for epoch in range(1, epochs + 1):
#     optimizer.zero_grad()
#     outputs = model(x)
#     loss = criterion(outputs, y)
#     loss.backward()
#     optimizer.step()

#     if epoch % 50 == 0 or epoch == 1:
#         print(f"📅 Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

# print("\n✅ Training Completed!")

# # Step 7: Inference Function
# def generate_text(start_char, max_length=20):
#     print(f"\n🧠 Starting generation with '{start_char}'...\n")
#     model.eval()

#     idx_input = torch.tensor([char_to_idx[start_char]], dtype=torch.long)
#     generated = [start_char]

#     for _ in range(max_length):
#         output_logits = model(idx_input)
#         probs = torch.softmax(output_logits[-1], dim=0)  # get the last token's probabilities
#         predicted_idx = torch.argmax(probs).item()

#         predicted_char = idx_to_char[predicted_idx]
#         generated.append(predicted_char)

#         # New input is the predicted character
#         idx_input = torch.cat([idx_input, torch.tensor([predicted_idx])])

#     final_text = ''.join(generated)
#     print(f"📝 Generated Text: {final_text}\n")
#     return final_text

# # Step 8: Test Generation
# generate_text('h', max_length=20)
# generate_text('l', max_length=20)
# generate_text('w', max_length=20)

import torch
import torch.nn as nn
import torch.optim as optim

print("\n🔹 Step 6: Inference and Text Generation 🔹\n")

# ----------------------------
# Step 1: Hyperparameters
# ----------------------------
vocab = sorted(set("hello world"))  # Unique characters sorted
vocab_size = len(vocab)
embedding_dim = 16
learning_rate = 0.01
epochs = 300  # Train for more epochs for better results

print(f"🔠 Vocab (Sorted Unique Characters): {vocab}")
print(f"🔢 Vocab Size: {vocab_size}")
print(f"📏 Embedding Dimension: {embedding_dim}")
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

# Encode "hello" into token IDs
encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)

# Prepare input x (except last token) and target y (shifted by 1)
x = encoded[:-1]   # Input: [h, e, l, l]
y = encoded[1:]    # Target: [e, l, l, o]

print(f"\n🧩 Encoded Input Tokens (x): {x}")
print(f"🎯 Encoded Target Tokens (y): {y}")

# ----------------------------
# Step 4: Define TinyTransformer Model
# ----------------------------
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Token Embedding: ID → 16D vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear Projection: 16D vector → vocab_size logits
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x_embed = self.embedding(x)    # [batch_size, embedding_dim]
        x_out = self.fc(x_embed)        # [batch_size, vocab_size]
        return x_out

# Instantiate model
model = TinyTransformer()

# ----------------------------
# Step 5: Loss and Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Step 6: Training Loop
# ----------------------------
print("\n🚀 Training the Model...\n")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    outputs = model(x)        # Forward pass
    loss = criterion(outputs, y)  # Loss calculation

    loss.backward()           # Backward pass (compute gradients)
    optimizer.step()           # Update parameters

    if epoch % 50 == 0 or epoch == 1:
        print(f"📅 Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

print("\n✅ Training Completed Successfully!")

# ----------------------------
# Step 7: Inference Function (Generate Text)
# ----------------------------
def generate_text(start_char, max_length=20):
    print(f"\n🧠 Starting generation from '{start_char}'...\n")
    model.eval()  # Set model to evaluation mode

    idx_input = torch.tensor([char_to_idx[start_char]], dtype=torch.long)
    generated = [start_char]  # Starting character

    for _ in range(max_length):
        output_logits = model(idx_input)
        
        # Focus on last token's output
        probs = torch.softmax(output_logits[-1], dim=0)
        
        # Choose token with highest probability (greedy decoding)
        predicted_idx = torch.argmax(probs).item()
        predicted_char = idx_to_char[predicted_idx]

        generated.append(predicted_char)

        # Prepare new input by appending predicted token
        idx_input = torch.cat([idx_input, torch.tensor([predicted_idx])])

    final_text = ''.join(generated)
    print(f"📝 Generated Text: {final_text}\n")
    return final_text

# ----------------------------
# Step 8: Test Text Generation
# ----------------------------
generate_text('h', max_length=20)
generate_text('l', max_length=20)
generate_text('w', max_length=20)

# ----------------------------
# Quick Concept Recap
# ----------------------------
print("""
🔵 Quick Concept Recap:
- We trained a tiny Transformer on "hello" text.
- Model learned to predict next characters sequentially.
- During generation:
  - Start with a character.
  - Predict next token.
  - Append and continue.
- We used greedy decoding (always picking most probable next token).

In real GPT models, sampling (top-k, temperature) is used for diversity!
""")
