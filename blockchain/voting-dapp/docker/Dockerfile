FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install py-solc-x web3
CMD ["python", "compile.py"]
