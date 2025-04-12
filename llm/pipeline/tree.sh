
#!/bin/bash

mkdir -p llm-pipeline/step1_train_custom/data
touch llm-pipeline/step1_train_custom/Dockerfile
touch llm-pipeline/step1_train_custom/docker-compose.yml
touch llm-pipeline/step1_train_custom/train.py
touch llm-pipeline/step1_train_custom/data/tiny_corpus.txt

mkdir -p llm-pipeline/step2_inference_custom
touch llm-pipeline/step2_inference_custom/Dockerfile
touch llm-pipeline/step2_inference_custom/docker-compose.yml
touch llm-pipeline/step2_inference_custom/generate.py

mkdir -p llm-pipeline/step3_finetune_custom/data
touch llm-pipeline/step3_finetune_custom/Dockerfile
touch llm-pipeline/step3_finetune_custom/docker-compose.yml
touch llm-pipeline/step3_finetune_custom/finetune.py
touch llm-pipeline/step3_finetune_custom/data/fine_tune_corpus.txt

mkdir -p llm-pipeline/step4_inference_finetuned
touch llm-pipeline/step4_inference_finetuned/Dockerfile
touch llm-pipeline/step4_inference_finetuned/docker-compose.yml
touch llm-pipeline/step4_inference_finetuned/generate_finetuned.py

mkdir -p llm-pipeline/step5_convert_gguf
touch llm-pipeline/step5_convert_gguf/Dockerfile
touch llm-pipeline/step5_convert_gguf/docker-compose.yml
touch llm-pipeline/step5_convert_gguf/convert_to_gguf.py

mkdir -p llm-pipeline/step6_quantize_gguf
touch llm-pipeline/step6_quantize_gguf/Dockerfile
touch llm-pipeline/step6_quantize_gguf/docker-compose.yml
touch llm-pipeline/step6_quantize_gguf/quantize.sh

mkdir -p llm-pipeline/step7_ollama_deploy
touch llm-pipeline/step7_ollama_deploy/Modelfile
touch llm-pipeline/step7_ollama_deploy/docker-compose.yml
touch llm-pipeline/step7_ollama_deploy/README.md

mkdir -p llm-pipeline/step8_convert_mlx
touch llm-pipeline/step8_convert_mlx/Dockerfile
touch llm-pipeline/step8_convert_mlx/docker-compose.yml
touch llm-pipeline/step8_convert_mlx/convert_to_mlx.py

echo "✅ llm-pipeline directory structure created successfully."

