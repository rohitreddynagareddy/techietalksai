# Use a lightweight base for Python on arm64
FROM python:3.13-slim
RUN pip install --upgrade pip && pip install torch \
torchvision torchaudio transformers datasets safetensors \
streamlit accelerate hf_xet
#RUN pip install torch torchvision torchaudio transformers datasets safetensors
=======

# Install necessary packages
RUN pip install --upgrade pip && pip install torch torchvision torchaudio transformers datasets safetensors \
streamlit accelerate hf_xet


# RUN pip install --upgrade pip && \
#     pip install \
#       torch==2.0.1 \
#       transformers==4.32.1 \
#       datasets==2.13.1 \
#       streamlit

# Xet Storage is enabled for this repo, but the 'hf_xet' 
# package is not installed. Falling back to regular HTTP download. 
# For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`


=======
RUN pip install torch torchvision torchaudio transformers datasets safetensors

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
=======
=======
# Command to run training
CMD ["python", "train.py"]
