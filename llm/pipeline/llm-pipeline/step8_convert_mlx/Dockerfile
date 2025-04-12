# step8_convert_mlx/Dockerfile
FROM python:3.13-slim
RUN pip install mlx-lm==0.21.5  # specify a version known to work
WORKDIR /app
COPY convert_to_mlx.py .
CMD ["python", "convert_to_mlx.py"]
