# step2_inference_custom/Dockerfile
FROM python:3.13-slim
RUN pip install torch transformers safetensors
WORKDIR /app
COPY generate.py .
# We assume the model will be mounted at /app/model
ENTRYPOINT ["python"]
CMD ["generate.py"]
