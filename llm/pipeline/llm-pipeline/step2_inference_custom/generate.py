import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load model and tokenizer
model_dir = "/app/model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

# Step 2: Device selection (MPS or CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("🚀 Using device:", device)
model.to(device)

# Step 3: Define the prompt
# prompt = "Hello, how"
prompt = "I love"
print("🧠 Prompt:", prompt)

inputs = tokenizer(prompt, return_tensors='pt').to(device)
print("🧩 Tokenized input_ids:", inputs["input_ids"])
print("🧩 Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# Step 4: Generate continuation
with torch.no_grad():
    # outputs = model.generate(**inputs, max_new_tokens=10)
    outputs = model.generate(**inputs, max_new_tokens=1)

print("📤 Generated token IDs:", outputs[0])
print("📜 Decoded output:", tokenizer.decode(outputs[0]))
