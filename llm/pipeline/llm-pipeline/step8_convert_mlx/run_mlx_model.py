from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Tinyllama-v0-1.1B-Chat-v1.0-4bit")
response = generate(model, tokenizer, "Who are you?")
print(response)


