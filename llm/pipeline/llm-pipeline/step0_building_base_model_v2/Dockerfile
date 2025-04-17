# Use an official Python runtime (ARM64) for Apple Silicon
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (nice for logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only pip requirements first so we leverage Docker cache
COPY requirements.txt .

# Install dependencies (CPU build of PyTorch)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Default command: train the model
CMD ["python", "train.py"]

