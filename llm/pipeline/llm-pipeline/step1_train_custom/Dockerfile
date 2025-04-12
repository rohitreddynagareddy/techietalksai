# Use a lightweight base for Python on arm64
FROM python:3.13-slim

# Install necessary packages
RUN pip install torch torchvision torchaudio transformers datasets safetensors

# Copy training script and data
WORKDIR /app
COPY train.py .
COPY data/ ./data/

# Set environment to avoid any interactive prompts
ENV PYTHONUNBUFFERED=1

# Command to run training
CMD ["python", "train.py"]
