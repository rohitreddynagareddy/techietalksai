python3 -m venv cpu_gpu
source cpu_gpu/bin/activate

# pip install --upgrade pip
# pip install --no-cache-dir -r requirements.txt


# cat > cpu-gpu.py <<EOT
# import torch
# print("MPS Available:", torch.backends.mps.is_available())
# print("MPS Built:", torch.backends.mps.is_built())
# EOT

python cpu-gpu.py

# python train.py
python generate.py

