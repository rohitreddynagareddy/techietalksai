# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size, --upgrade pip ensures we have the latest pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (the 'app' directory and its contents)
# into the container at /app
COPY ./app /app/app

# Make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

# Define environment variable to ensure python outputs everything without buffering
ENV PYTHONUNBUFFERED 1

# Command to run the Streamlit application
# Use 0.0.0.0 to make it accessible from outside the container
# Use --server.enableCORS=false for simpler setups if needed, but usually default is fine
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
