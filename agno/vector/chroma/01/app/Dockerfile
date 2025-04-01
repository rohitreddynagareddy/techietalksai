FROM python:3.11-slim

#FROM python:3.9-slim-bookworm

WORKDIR /app

# First update package lists and install security updates
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Then install build dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Add these before pip install
RUN apt-get update && \
    apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
