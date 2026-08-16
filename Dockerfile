FROM python:3.11-slim

# Prevent Python buffering and direct YOLO config to /tmp
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV PORT=8000

# Install system packages required by OpenCV, FFmpeg and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install CPU PyTorch lightweight wheels
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained assets
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.app:create_api_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
