# step3_finetune_custom/finetune.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model and tokenizer (from mounted volume)
model_dir = "/app/base_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Load new fine-tuning text
data_path = "data/fine_tune_corpus.txt"
with open(data_path, 'r') as f:
    new_text = f.read()
# Tokenize new text
inputs = tokenizer(new_text, return_tensors='pt')
input_ids = inputs["input_ids"]

# Move model to MPS or CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.train()

# Simple fine-tuning loop (few epochs)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # lower LR for fine-tuning
epochs = 3
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(input_ids.to(device), labels=input_ids.to(device))
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Fine-tune Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save fine-tuned model
os.makedirs("output_finetuned", exist_ok=True)
model.save_pretrained("output_finetuned")
tokenizer.save_pretrained("output_finetuned")
print("Fine-tuned model saved to output_finetuned/")
