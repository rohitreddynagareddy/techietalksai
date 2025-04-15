import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# 1. Load and tokenize your text data
print("🔹 Step 1: Loading the dataset...")
data_path = "data/tiny_corpus.txt"
with open(data_path, 'r') as f:
    text = f.read()
print("\n📘 Raw text sample:\n", text[:200], "...")

print("\n🔹 Step 2: Loading tokenizer and encoding text...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs["input_ids"]
print("\n🧠 Token IDs:\n", input_ids)
print("📏 Shape of input_ids:", input_ids.shape)

decoded = tokenizer.decode(input_ids[0])
print("\n🔁 Decoded back from tokens:\n", decoded)

# 2. Define a small GPT-2 model configuration
print("\n🔹 Step 3: Creating a small GPT-2 model...")
config = AutoConfig.from_pretrained("gpt2")
config.n_layer = 2
config.n_head = 2
config.n_embd = 128
config.vocab_size = tokenizer.vocab_size

model = AutoModelForCausalLM.from_config(config)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.train()
print("📦 Model loaded on:", device)

# 3. Training Loop
print("\n🔹 Step 4: Starting training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
epochs = 5

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(input_ids.to(device), labels=input_ids.to(device))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"✅ Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 4. Save model and tokenizer
print("\n🔹 Step 5: Saving model and tokenizer to `output_model/` ...")
os.makedirs("output_model", exist_ok=True)
model.save_pretrained("output_model")
tokenizer.save_pretrained("output_model")
print("📁 Saved files:")
for file in os.listdir("output_model"):
    print("   -", file)

# 5. Extra: Peek inside config and model files
print("\n🔍 Peek into config.json:")
with open("output_model/config.json") as f:
    print(f.read()[:500], "...")

print("\n✅ Done! Your trained model is ready in the 'output_model/' folder.")
