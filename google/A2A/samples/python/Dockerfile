# Use the official Python base image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app


# Install system dependencies including curl
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (universal virtual environment manager)
RUN pip install --upgrade pip && pip install uv

# Copy project files
COPY . /app/

# Install Python dependencies using uv
# RUN uv pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Default command
CMD ["python", "hosts/cli/main.py"]

