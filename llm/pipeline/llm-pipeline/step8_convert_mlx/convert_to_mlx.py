# convert_to_mlx.py
from mlx_lm import load, generate

# You can replace this with your own model if it's been quantized + converted
# model_id = "mlx-community/Tinyllama-v0-1.1B-Chat-v1.0-4bit"
# model_id = "mlx-community/phi-2-mlx"
model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"


print(f"🔁 Loading model: {model_id}")
model, tokenizer = load(model_id)

prompt = "What is the purpose of life?"
print(f"\n🧠 Prompt: {prompt}")
response = generate(model, tokenizer, prompt)

print("\n💬 Response:")
print(response)
