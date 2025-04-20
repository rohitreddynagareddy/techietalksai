# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app code
COPY app.py .

# Install dependencies
RUN pip install streamlit graphviz

# Expose port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

