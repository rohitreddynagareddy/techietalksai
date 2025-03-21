FROM python:3.11-slim
WORKDIR /app
COPY echo_server.py .
RUN pip install fastmcp
RUN pip install starlette uvicorn mcp
RUN pip install fastapi
EXPOSE 8000
# CMD ["python", "echo_server1.py"]

