import subprocess
import os

hf_model_path = "/app/ft_model"
output_path = "output_gguf/model-f16.gguf"

print(f"Converting model from: {hf_model_path} to: {output_path}")

# ✅ Clone llama.cpp only if not already cloned
if not os.path.exists("llama.cpp"):
    print("📦 Cloning llama.cpp...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)
else:
    print("✅ llama.cpp already exists. Skipping clone.")

# ✅ Install dependencies
subprocess.run(["pip", "install", "--no-cache-dir", "transformers", "safetensors"], check=True)

# ✅ Convert HF model to GGUF
print("⚙️ Running convert_hf_to_gguf.py...")
subprocess.run([
    "python", "llama.cpp/convert_hf_to_gguf.py", hf_model_path,
    "--outfile", output_path
], check=True)

print("✅ GGUF conversion complete.")
