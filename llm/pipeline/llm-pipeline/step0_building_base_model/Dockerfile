FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install torch transformers tqdm

CMD ["python", "train_base_model.py"]
