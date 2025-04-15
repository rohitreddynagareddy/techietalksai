# step1_train_custom/train.py
import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# 1. Prepare dataset
data_path = "data/tiny_corpus.txt"
with open(data_path, 'r') as f:
    text = f.read()
# Simple preprocessing: split into lines and tokenize
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # use pre-trained GPT2 tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # pad token needed for GPT2

# Encode the entire text into tokens (for tiny dataset, we do a simple approach)
inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs["input_ids"]

# 2. Define tiny model config (GPT-2 architecture with reduced size)
config = AutoConfig.from_pretrained("gpt2")
config.n_layer = 2            # 2 transformer decoder layers
config.n_head = 2            # 2 attention heads
config.n_embd = 128          # embedding dimension
config.vocab_size = tokenizer.vocab_size
model = AutoModelForCausalLM.from_config(config)

# Use Apple MPS if available for speed
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.train()

# 3. Training loop (very basic)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
epochs = 5
for epoch in range(epochs):
    # For simplicity, feed the whole corpus as one sequence (tiny dataset)
    optimizer.zero_grad()
    outputs = model(input_ids.to(device), labels=input_ids.to(device))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 4. Save model & tokenizer in Hugging Face format
os.makedirs("output_model", exist_ok=True)
model.save_pretrained("output_model")
tokenizer.save_pretrained("output_model")
print("Model saved to output_model/")
