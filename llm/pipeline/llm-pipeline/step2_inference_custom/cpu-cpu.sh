
python3 -m venv cpu_gpu
source cpu_gpu/bin/activate

pip install --upgrade pip
pip install torch numpy


cat > cpu-gpu.py <<EOT
import torch
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())
EOT

python3 cpu-gpu.py

# MPS Available: True
# MPS Built: True