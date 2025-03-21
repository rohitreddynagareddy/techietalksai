FROM python:3.11-slim
WORKDIR /app
COPY client.py .
RUN pip install mcp fastmcp
# RUN pip install starlette uvicorn mcp
RUN pip install mcp aiohttp

CMD ["python", "client.py"]

