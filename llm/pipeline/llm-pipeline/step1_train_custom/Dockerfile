# Use a lightweight base for Python on arm64
FROM python:3.13-slim
RUN pip install --upgrade pip && pip install torch \
torchvision torchaudio transformers datasets safetensors \
streamlit accelerate hf_xet
#RUN pip install torch torchvision torchaudio transformers datasets safetensors

# Copy training script and data
WORKDIR /app
COPY train.py .
COPY data/ ./data/

# Set environment to avoid any interactive prompts
ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# Command to run training
# CMD ["python", "train.py"]
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
#CMD ["python", "train.py"]
