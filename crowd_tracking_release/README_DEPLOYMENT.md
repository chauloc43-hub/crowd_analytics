# 🚀 BỘ HƯỚNG DẪN CÀI ĐẶT & VẬN HÀNH DỰ ÁN CROWD TRACKING (TỪ A - Z)

Tài liệu này hướng dẫn chi tiết cách cài đặt, tải các mô hình AI weights và vận hành ứng dụng **Crowd Tracking & AI Gender Analytics** từ repo GitHub.

---

## 📋 YÊU CẦU HỆ THỐNG
- **Python**: 3.10 hoặc 3.11 (Khuyên dùng Python 3.10).
- **Git**: Đã cài đặt trên hệ thống.
- **Trình duyệt**: Chrome, Safari, Edge hoặc Firefox (hỗ trợ WebRTC & HTTPS Webcam).

---

## 🛠️ BƯỚC 1: CLONE DỰ ÁN VỀ MÁY TÍNH
Mở Terminal / Command Prompt và chạy:

```bash
# Clone dự án từ GitHub
git clone https://github.com/Qorynx/crowd_tracking.git

# Di chuyển vào thư mục dự án
cd crowd_tracking
```

---

## 🐍 BƯỚC 2: TẠO VÀ KÍCH HOẠT MÔI TRƯỜNG ẢO

### Trên macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Trên Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
# Hoặc trên PowerShell: .\.venv\Scripts\Activate.ps1
```

---

## 📦 BƯỚC 3: CÀI ĐẶT THƯ VIỆN PHỤ THUỘC (DEPENDENCIES)

```bash
# Nâng cấp pip lên bản mới nhất
python -m pip install --upgrade pip

# Cài đặt thư viện xử lý chính và API runtime
pip install -r requirements.txt
pip install -r deploy/requirements-api-runtime.txt
```

---

## 🤖 BƯỚC 4: TẢI CÁC MÔ HÌNH AI (MODEL WEIGHTS)

Tạo cấu trúc thư mục và tải các file trọng số AI chính chủ từ GitHub Release:

```bash
# 1. Tạo các thư mục chứa model
mkdir -p artifacts/person_detector artifacts/face_detector artifacts/gender_classifier artifacts/body_gender_classifier

# 2. Tải YOLO11n (Person Detector)
curl -L "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" \
  -o "artifacts/person_detector/yolo11n.pt"

# 3. Tải YuNet Face Detector
curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  -o "artifacts/face_detector/face_detection_yunet_2023mar.onnx"

# 4. Tải Face Gender Classifier
curl -L "https://github.com/Qorynx/crowd_tracking/releases/download/v0.1.0/face_gender_classifier_mobilenet_v3_large.pth" \
  -o "artifacts/gender_classifier/best_model.pth"

# 5. Tải Body Gender Classifier
curl -L "https://github.com/Qorynx/crowd_tracking/releases/download/v0.1.0/body_gender_classifier_mobilenet_v3_small.pth" \
  -o "artifacts/body_gender_classifier/best_body_gender_model.pth"

# 6. Kiểm tra tính hợp lệ của toàn bộ mô hình AI đã tải
python tools/prepare_production_assets.py
```

---

## 🔑 BƯỚC 5: TẠO CHỨNG CHỈ HTTPS SSL (BẮT BUỘC ĐỂ DÙNG WEBCAM / ĐIỆN THOẠI)

Trình duyệt hiện đại bắt buộc kết nối **HTTPS** thì mới cấp quyền mở Camera trên cả máy tính lẫn điện thoại:

```bash
openssl req -x509 -newkey rsa:2048 -keyout cert.key -out cert.crt -days 365 -nodes -subj "/CN=localhost"
```

---

## ⚡ BƯỚC 6: CHẠY DỰ ÁN

### 🌟 Cách 1: Chạy bản Cyber-HUD Luxury Web App (Khuyên dùng - Có HTTPS & Nhận diện Giới tính)
```bash
uvicorn src.api.app:create_api_app --factory --host 0.0.0.0 --port 8000 --ssl-keyfile cert.key --ssl-certfile cert.crt
```

### 🎛️ Cách 2: Chạy bản Giao diện Thử nghiệm Gradio
```bash
python app.py
```

---

## 🌐 BƯỚC 7: TRUY CẬP VÀ TRẢI NGHIỆM

1. **Trên máy tính local**: Mở trình duyệt gõ: **`https://localhost:8000`**
2. **Trên điện thoại chung WiFi**: 
   - Kiểm tra IP máy tính: `ipconfig getifaddr en0` (Mac) hoặc `ipconfig` (Windows).
   - Trên điện thoại gõ: **`https://<IP_MAY_TINH>:8000`** *(Ví dụ: https://192.168.1.6:8000)*
3. **Thao tác SSL lần đầu**: Chọn **Advanced (Nâng cao)** ➔ Bấm **Proceed / Tiếp tục truy cập**.
4. Cấp quyền **Allow (Cho phép)** dùng Camera ➔ Bấm **`▶ BẮT ĐẦU STREAM`**.

---

## 🛠️ XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

| Lỗi gặp phải | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `[Errno 48] Address already in use` | Cổng 8000 đang bị tiến trình cũ chiếm | Chạy lệnh `lsof -ti:8000 \| xargs kill -9` trên Mac/Linux để giải phóng cổng. |
| `Gender checkpoint not found` | Chưa tải file model AI hoặc sai tên file | Chạy lại lệnh ở **Bước 4** để tải đủ 4 file weights vào thư mục `artifacts/`. |
| Giao diện không mở được Camera trên Điện thoại | Truy cập bằng HTTP hoặc chưa cấp quyền SSL | Đảm bảo đường dẫn truy cập có tiền tố **`https://`**. |
