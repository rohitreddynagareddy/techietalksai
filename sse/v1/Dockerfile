# Use Node.js 23 as the base image
FROM node:23

# Set the working directory
WORKDIR /app

# Install Python 3.11 and required dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip python3-full \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure 'python' points to 'python3.11'
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Verify Python version
RUN python --version

# Create a virtual environment for Python
RUN python -m venv /app/venv

# Ensure the virtual environment is activated in future commands
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip inside the virtual environment
RUN /app/venv/bin/pip install --upgrade pip setuptools wheel

# Install uv (fast Python package installer)
RUN /app/venv/bin/pip install uv

# Copy the requirements file before installing dependencies
COPY requirements.txt .

# Install dependencies inside the virtual environment
# RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt
# Install dependencies using uv
RUN /app/venv/bin/uv pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application files
COPY . .

# Expose necessary ports
EXPOSE 8501

# Set the correct shell for subsequent commands
SHELL ["/bin/bash", "-c"]

# Verify that dependencies are installed
RUN /app/venv/bin/pip list

# Set the working directory
WORKDIR /app/app

# ✅ FIX: Correct ENTRYPOINT and CMD
#ENTRYPOINT ["python"]
#CMD ["filesystem.py"]

