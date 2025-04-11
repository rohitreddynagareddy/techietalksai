FROM python:3.11-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY multi_tool_agent ./multi_tool_agent
COPY entrypoint.sh .

# Run web interface by default
ENTRYPOINT ["bash", "entrypoint.sh"]

