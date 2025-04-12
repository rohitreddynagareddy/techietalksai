# convert_to_mlx.py

from mlx_lm import load, generate

model_path = "./output_mlx"

print(f"🔁 Loading MLX model from: {model_path}")
model, tokenizer = load(model_path)

prompt = "Hello, how are you?"
print(f"\n🧠 Prompt: {prompt}")
response = generate(model, tokenizer, prompt=prompt)

print("\n💬 Response:")
print(response)

