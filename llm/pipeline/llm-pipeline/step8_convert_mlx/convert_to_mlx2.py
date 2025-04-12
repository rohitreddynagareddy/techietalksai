# step8_convert_mlx/convert_to_mlx.py
import mlx_lm

# Path or name of HF model to convert. 
hf_model_path = "/app/ft_model"  # using the fine-tuned model from Step 3 as an example
output_dir = "/app/mlx_model"    # where to save the MLX-converted model

# Convert the model to MLX format with quantization
mlx_lm.convert(
    hf_path=hf_model_path,
    mlx_path=output_dir,
    quantize=True  # enable quantization (by default, MLX uses 4-bit quantization for supported models)
)
print("MLX conversion complete. Saved to:", output_dir)
