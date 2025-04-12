# step5_convert_gguf/Dockerfile
FROM python:3.13-slim

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install required Python libraries
RUN pip install --no-cache-dir \
    torch \
    transformers \
    safetensors \
    sentencepiece

WORKDIR /app

COPY convert_to_gguf.py .

CMD ["python", "convert_to_gguf.py"]
