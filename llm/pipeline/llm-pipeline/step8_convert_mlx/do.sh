
# pip install mlx-lm

# https://huggingface.co/models?library=mlx

cd ../step8_convert_mlx/
python3 -m venv mlx_env
source mlx_env/bin/activate

pip install --upgrade pip
pip install mlx-lm

pip install -U huggingface_hub
huggingface-cli login

python convert_to_mlx.py
python hf.py



