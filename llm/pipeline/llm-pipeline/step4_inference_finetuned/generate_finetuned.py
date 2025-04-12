# step4_inference_finetuned/generate_finetuned.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "/app/ft_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

prompt = "I am a tiny model"
inputs = tokenizer(prompt, return_tensors='pt').to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5)
print("Prompt:", prompt)
print("Continuation:", tokenizer.decode(outputs[0]))
