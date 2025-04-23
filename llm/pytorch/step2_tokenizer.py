import torch

print("\n🔹 Step 2: Tiny Dataset and Tokenizer 🔹\n")

# Tiny corpus
corpus = "I am Sree from Schogini.com"
print(f"📚 Tiny Corpus: {corpus}")

# Create vocabulary (set of unique characters)
vocab = sorted(set(corpus))
print(f"🔤 Vocabulary: {vocab}")

# Mappings
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

print(f"🔢 char_to_idx: {char_to_idx}")
print(f"🔢 idx_to_char: {idx_to_char}")

# Encode corpus to integers
encoded = torch.tensor([char_to_idx[ch] for ch in corpus], dtype=torch.long)
print(f"🧩 Encoded Corpus: {encoded}")

# Decode back
decoded = ''.join([idx_to_char[idx.item()] for idx in encoded])
print(f"🔁 Decoded Corpus: {decoded}")
