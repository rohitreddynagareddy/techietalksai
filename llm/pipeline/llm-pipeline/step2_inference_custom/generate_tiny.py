import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load model and tokenizer
model_dir = "./tiny_gpt2_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

# Step 2: Device selection (MPS or CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("🚀 Using device:", device)
model.to(device)

# Step 3: Define the prompt
prompt = "Natural Language Processing enables"
print("🧠 Prompt:", prompt)

inputs = tokenizer(prompt, return_tensors='pt').to(device)
print("🧩 Tokenized input_ids:", inputs["input_ids"])
print("🧩 Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# Step 4: Generate continuation
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        top_k=50,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

print("📤 Generated token IDs:", outputs[0])
print("📜 Decoded output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
