# step2_inference_custom/Dockerfile
FROM python:3.13-slim
RUN pip install torch transformers safetensors
WORKDIR /app
COPY generate.py .
# We assume the model will be mounted at /app/model
<<<<<<< HEAD
ENTRYPOINT ["python"]
CMD ["generate.py"]
=======
CMD ["python", "generate.py"]
>>>>>>> cb7769eceecd169c19534ed4e09ae8b48ac3e8a8
