FROM python:3.11-slim

WORKDIR /app

# Added ffmpeg - Faster-Whisper WILL fail without it
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the environment variable so it can find 'Backend'
ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 7860

ENV PYTHONPATH="/app:/app/Backend"

# We run from /app so the 'Backend' and 'Frontend' folders are visible
CMD ["sh", "-c", "python3 -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000 & python3 Frontend/app.py"]