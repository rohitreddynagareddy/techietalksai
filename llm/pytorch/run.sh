
[ $# -ne 1 ] && exho Need Python Script && exit

python3 -m venv ~/py-gpu-mlx
source ~/py-gpu-mlx/bin/activate

pip install torch numpy
pip install -U huggingface_hub
pip install torch numpy transformers \
datasets tqdm transformers[torch] accelerate \
hf_xet tiktoken blobfile

#huggingface-cli login

python $1
