# Dockerfile
FROM python:3.11-slim

# Install pip packages
RUN pip install torch numpy transformers \
datasets tqdm transformers[torch] accelerate \
hf_xet tiktoken blobfile
# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Default command
CMD ["python", "step1_env_check.py"]

