# step3_finetune_custom/Dockerfile
FROM python:3.13-slim
RUN pip install torch transformers safetensors
WORKDIR /app
COPY finetune.py .
COPY data/ ./data/
CMD ["python", "finetune.py"]
