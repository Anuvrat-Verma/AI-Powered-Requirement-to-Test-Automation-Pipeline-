# Use a Python 3.11 base image
FROM python:3.11-slim

# Install system dependencies for Faster-Whisper and Selenium
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

# Expose the ports for FastAPI and Gradio
EXPOSE 8000
EXPOSE 7860

# Run both the backend and frontend
CMD ["sh", "-c", "uvicorn Backend.main:app --host 0.0.0.0 --port 8000 & python Frontend/app.py"]