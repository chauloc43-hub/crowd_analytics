FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download production AI model weights during Docker build on Hugging Face
RUN mkdir -p artifacts/person_detector artifacts/face_detector artifacts/gender_classifier artifacts/body_gender_classifier && \
    curl -L "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" -o artifacts/person_detector/yolo11n.pt && \
    curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -o artifacts/face_detector/face_detection_yunet_2023mar.onnx && \
    curl -L "https://github.com/Qorynx/crowd_tracking/releases/download/v0.1.0/face_gender_classifier_mobilenet_v3_large.pth" -o artifacts/gender_classifier/face_gender_classifier_mobilenet_v3_large.pth && \
    curl -L "https://github.com/Qorynx/crowd_tracking/releases/download/v0.1.0/body_gender_classifier_mobilenet_v3_small.pth" -o artifacts/body_gender_classifier/body_gender_classifier_mobilenet_v3_small.pth

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "src.api.app:create_api_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
