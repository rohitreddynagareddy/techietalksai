
# pip install mlx-lm

# https://huggingface.co/models?library=mlx

cd ../step8_convert_mlx/
python3 -m venv mlx_env
source mlx_env/bin/activate

pip install --upgrade pip
pip install mlx-lm

# pip install -U huggingface_hub
# huggingface-cli login

# python -m mlx_lm.convert --hf-path mlx-community/phi-2 --mlx-path ./mlx_phi2 --quantize


python -m mlx_lm.convert \
  --hf-path ../step3_finetune_custom/output_finetuned/ \
  --mlx-path ./output_mlx \
  --quantize


# python convert_our_to_mlx.py

#python convert_to_mlx.py
#python hf.py



