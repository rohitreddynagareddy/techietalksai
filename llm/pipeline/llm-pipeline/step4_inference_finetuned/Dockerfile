# step4_inference_finetuned/Dockerfile
FROM python:3.13-slim
RUN pip install torch transformers safetensors
WORKDIR /app
COPY generate_finetuned.py .
CMD ["python", "generate_finetuned.py"]
