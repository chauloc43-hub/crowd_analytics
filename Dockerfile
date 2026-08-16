FROM python:3.11-slim

# Install system packages required by OpenCV, FFmpeg and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained assets
COPY . .

# Expose Hugging Face Space default port
EXPOSE 7860

# Run FastAPI server on port 7860
CMD ["uvicorn", "src.api.app:create_api_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
