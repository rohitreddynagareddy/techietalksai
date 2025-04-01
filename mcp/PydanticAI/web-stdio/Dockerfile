# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application context (including mcp_server and streamlit_app folders)
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Check if OPENAI_API_KEY is set at runtime (optional sanity check)
# RUN echo "Checking for OPENAI_API_KEY presence during build (won't have value yet): ${OPENAI_API_KEY:-Not Set}"

# Command to run the Streamlit application
# Streamlit will then launch the MCP server process via stdio_client
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
