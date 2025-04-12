# step5_convert_gguf/convert_to_gguf.py
import subprocess

# Parameters
hf_model_name = "TinyLlama-1.1B"            # huggingface model (change to local path if needed)
output_path = "model-f16.gguf"             # output GGUF file (FP16 full precision)
# Note: We will quantize later, so here we keep full precision.

# Clone llama.cpp (if not already in image)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)

# Install requirements for conversion
subprocess.run(["pip", "install", "transformers", "safetensors"], check=True)

# Run the conversion script
conv_script = "llama.cpp/convert_hf_to_gguf.py"
subprocess.run(["python", conv_script, hf_model_name, output_path], check=True)

print(f"Conversion to GGUF completed: {output_path}")
