# step2_inference_custom/generate.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer from the saved directory
model_dir = "/app/model"  # this will be a mounted volume
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()

# Use MPS if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Define a prompt for the model
prompt = "Hello, how"
inputs = tokenizer(prompt, return_tensors='pt').to(device)

# Generate continuation
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
generated_text = tokenizer.decode(outputs[0])
print(f"Prompt: {prompt}\nModel continuation: {generated_text}")
