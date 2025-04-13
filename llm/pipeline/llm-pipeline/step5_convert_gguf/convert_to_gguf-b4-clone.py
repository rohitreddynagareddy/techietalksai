import subprocess

hf_model_path = "/app/ft_model"
output_path = "output_gguf/model-f16.gguf"

print(f"Converting model from: {hf_model_path} to: {output_path}")

# Clone llama.cpp if not already present

# SREE COMMENTED
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)

# Install dependencies
subprocess.run(["pip", "install", "--no-cache-dir", "transformers", "safetensors"], check=True)

# Run the correct script
subprocess.run([
    # "python", "llama.cpp/convert-hf-to-gguf.py", hf_model_path,
    "python", "llama.cpp/convert_hf_to_gguf.py", hf_model_path,
    "--outfile", output_path
], check=True)

print("✅ GGUF conversion complete.")
