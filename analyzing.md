# Phân tích Bottleneck Hiệu Năng — Crowd Analytics MVP

Tài liệu này phân tích các điểm nghẽn (bottleneck) hiện tại trong pipeline xử lý
(`src/inference/`, `src/tracking/`, `src/analytics/`, `src/api/`) dựa trên việc
đọc trực tiếp source code, và đề xuất hướng cải thiện cho từng điểm. Không bao
gồm `.venv`, dataset, hoặc model binaries.

## 1. Kiến trúc pipeline hoàn toàn tuần tự (single-thread, single-stream)

**Vị trí:** `src/inference/pipeline.py::process_frame`, `src/inference/runtime.py::ModelRuntime`

Một frame đi qua các giai đoạn hoàn toàn tuần tự trên **một thread duy nhất**:
`tracking -> face extraction (loop) -> face classify batch -> body extraction (loop)
-> body classify batch -> analytics -> drawing`. Không có pipelining giữa các
giai đoạn (ví dụ: GPU chạy detector cho frame N+1 trong khi CPU đang vẽ overlay
cho frame N).

`ModelRuntime` được thiết kế **chỉ phục vụ đúng 1 stream** vì Ultralytics gắn
tracker state vào `YOLO.track()` — comment trong code xác nhận điều này
("one ModelRuntime must therefore serve only one active webcam/video stream").
Hệ quả: mỗi stream/session phải load riêng một bộ YOLO + YuNet + 2 classifier
vào VRAM, không thể chia sẻ trọng số hoặc batch detector/classifier qua nhiều
stream cùng lúc.

**Ảnh hưởng:** Thông lượng của toàn hệ thống bị giới hạn bởi latency của
**một** pipeline; không tận dụng được GPU khi có nhiều camera vì mỗi camera là
một tiến trình suy luận độc lập, không share compute.

**Cải thiện có thể:**
- Tách detector/tracker khỏi việc gắn cứng vào `YOLO.track()` (viết adapter
  tracking riêng) để nhiều stream có thể **share một bộ trọng số detector** và
  gộp batch detection của nhiều camera trong một forward pass.
- Áp dụng producer/consumer pipeline có hàng đợi giữa các stage (detect →
  attribute-extract → classify → analytics/draw) chạy trên các thread/luồng CUDA
  stream khác nhau để overlap GPU và CPU.

## 2. Trích xuất khuôn mặt/thân thể theo vòng lặp tuần tự, không batch hóa ở bước detect

**Vị trí:** `src/inference/pipeline.py::_collect_gender_candidates`,
`src/inference/runtime.py::extract_gender_candidate_result`,
`extract_body_gender_candidate`

Với mỗi track được lên lịch (tối đa `max_face_tracks_per_frame=4`,
`max_body_tracks_per_frame=4`), YuNet được gọi **riêng lẻ từng người một**:
mỗi lần `self.yunet.setInputSize(...)` + `self.yunet.detect(...)` trên một ROI
crop khác nhau. Đây là các lệnh OpenCV chạy trên CPU, tuần tự trong Python
loop — không vectorize được vì API của `cv2.FaceDetectorYN` chỉ nhận 1 ảnh/lần.

**Ảnh hưởng:** Khi đông người (crowd), phần face-extraction (CPU-bound) có thể
trở thành nút cổ chai chiếm phần lớn `face_detection` trong `timing_ms`, đặc
biệt khi `setInputSize` phải resize lại buffer nội bộ của YuNet mỗi lần gọi.

**Cải thiện có thể:**
- Gom các ROI thành một canvas lớn (grid packing) và gọi YuNet một lần thay vì
  N lần, nếu độ chính xác chấp nhận được.
- Cache `setInputSize` nếu nhiều crop cùng kích thước liên tiếp, tránh việc
  gọi lại API resize nội bộ không cần thiết.
- Xem xét thay YuNet bằng một face detector GPU-batch được (ví dụ ONNXRuntime
  GPU với batch input cố định).

## 3. Đồng bộ hoá CPU↔GPU (`.cpu()`/`.numpy()`) lặp lại mỗi frame

**Vị trí:** `src/inference/runtime.py::active_tracks`,
`_capture_pre_tracker_detections`, `classify_gender_batch`,
`classify_body_gender_batch`

Mỗi frame đều có ít nhất các lệnh:
```
results[0].boxes.xyxy.cpu().numpy()
results[0].boxes.id.cpu().numpy()
```
và trong `classify_gender_batch`/`classify_body_gender_batch`:
```
logits.float().cpu().numpy()
probabilities.max(dim=1).values.float().cpu().numpy()
```
Mỗi lệnh `.cpu()` là một điểm **đồng bộ hoá CUDA stream bắt buộc** — GPU phải
hoàn tất toàn bộ công việc đang chờ trước khi trả dữ liệu về host. Khi
`telemetry_enabled=true` (chẩn đoán detector), còn có thêm `_tensor_to_numpy`
copy cả `boxes.conf` và `boxes.xyxy` mỗi frame.

**Ảnh hưởng:** Trên GPU, các sync point này ngăn việc gối đầu (overlap) giữa
tính toán và truyền dữ liệu, làm giảm hiệu năng thực tế so với lý thuyết,
đặc biệt rõ khi batch nhỏ (1-4 track/frame — rất phổ biến ở webcam demo).

**Cải thiện có thể:**
- Gom nhiều lệnh `.cpu()` thành một lệnh transfer duy nhất (ví dụ `torch.cat`
  các tensor cần chuyển trước khi gọi `.cpu()` một lần).
- Giữ `telemetry_enabled=false` trong production (đã là default) và chỉ bật
  tạm thời khi calibrate.
- Dùng CUDA stream riêng cho hậu xử lý (non-blocking `.cpu()` với
  `pin_memory=True`) để giảm stall.

## 4. Cấu hình detector ưu tiên độ nhạy (recall) hơn tốc độ

**Vị trí:** `configs/pipeline-live.yaml` (`person_detector`), `runtime.py::_track_kwargs`

`tracker_input_confidence=0.05` (rất thấp) + `max_det=300` + NMS thường (không
`end2end`) khiến detector phải xử lý (postprocess/NMS) một lượng lớn candidate
box có điểm số thấp mỗi frame — đây là đánh đổi có chủ đích để FastTracker có
đủ "low-score recovery band", nhưng đồng nghĩa chi phí NMS cao hơn so với một
ngưỡng lọc chặt hơn.

Ngoài ra, cơ chế **detector recovery** (`recovery.enabled=true`) sẽ tự boost
`imgsz` từ 512 lên 640 trong `boost_frames=3` frame liên tiếp khi phát hiện
detector "rớt" — đây là chi phí bổ sung có chủ đích, nhưng nếu điều kiện camera
xấu kéo dài, hệ thống có thể liên tục dao động giữa 512/640 (bounded bởi
`cooldown_frames=45`), gây dao động độ trễ (latency jitter) khó lường.

**Cải thiện có thể:**
- Thử `end2end=true` (NMS-free head, nếu model hỗ trợ) để giảm chi phí
  postprocess.
- Theo dõi `runtime.detector.recovery.totals` trong production để xác định
  camera nào trigger recovery quá thường xuyên và điều chỉnh threshold/ánh
  sáng thay vì để hệ thống liên tục tự boost.

## 5. Tính toán/serialize lại toàn bộ heatmap + zone state 2 lần mỗi frame

**Vị trí:** `src/inference/pipeline.py::process_frame` (dòng gọi
`spatial_engine.update` rồi `spatial_engine.get_statistics()` lần 2 sau
`finalize_stale_tracks`), `src/analytics/spatial.py::get_statistics`

Trong một frame, `SpatialEngine.get_statistics()` (và tương tự
`classroom_analytics.get_statistics()`) được gọi **2 lần**: một lần ngầm trong
`update()`, một lần sau `finalize_stale_tracks()` để cập nhật dwell-time đã
hoàn tất. Hàm này build lại toàn bộ heatmap grid (mảng lồng Python, không dùng
`tolist()`/vectorized) bằng list-comprehension với `round()` từng phần tử:
```python
"values": [[round(float(value), 3) for value in row] for row in self._heatmap]
```
áp dụng cho cả `_heatmap` và `_accumulated_heatmap` — với grid mặc định 16x12,
đây là ~192 x 2 phép `round`/frame chỉ để phục vụ payload, cộng dồn với các
list/dict comprehension khác trong `zones`, `distribution`.

**Ảnh hưởng:** Đây là overhead Python thuần (không GPU), tăng tuyến tính theo
kích thước grid heatmap và số zone; xảy ra bất kể UI có hiển thị heatmap hay
không (trong khi `show_heatmap` chỉ ảnh hưởng tới việc vẽ overlay lên frame,
không ảnh hưởng tới việc build lại statistics dict).

**Cải thiện có thể:**
- Dùng `numpy.round(...).tolist()` (vectorized) thay cho list-comprehension
  từng phần tử — nhanh hơn đáng kể với numpy.
- Cache kết quả `get_statistics()` trong 1 frame và tái sử dụng thay vì gọi lại
  lần 2 sau `finalize_stale_tracks` khi không có track nào bị finalize.
- Cho phép API/Gradio yêu cầu heatmap ở độ chi tiết thấp hơn (bỏ field
  `"values"` khi client không cần), giảm payload lẫn CPU serialize.

## 6. `PersonIdentityResolver.update` có độ phức tạp gần bậc hai theo số người

**Vị trí:** `src/tracking/person_identity.py::update`

Với mỗi track chưa resolve (`unresolved`), thuật toán so khớp với **toàn bộ**
các person record còn "dormant" (`for person_id, record in self._records.items()`),
tính điểm số, rồi sort danh sách candidate cho cả track và person để tìm "clear
winner". Đây là vòng lặp lồng nhau O(unresolved_tracks × active_person_records)
chạy bằng Python thuần (không numpy) mỗi frame.

**Ảnh hưởng:** Với vài người (demo/webcam) chi phí này không đáng kể, nhưng
đây là điểm sẽ **scale kém** nếu mở rộng sang cảnh đông người thật (hàng chục
người) — đúng use-case "crowd analytics" mà tên project hướng tới.

**Cải thiện có thể:**
- Dùng spatial indexing (grid/k-d tree) để chỉ so khớp các track/person ở gần
  nhau về vị trí, tránh so toàn bộ N×M.
- Giới hạn số dormant record được xét đồng thời (đã có `max_inactive_frames`
  nhưng chưa giới hạn theo *số lượng*, chỉ theo *thời gian*).

## 7. Vẽ overlay (`_draw_results`) dùng loop OpenCV per-track thuần Python

**Vị trí:** `src/inference/pipeline.py::_draw_results`

Với mỗi track: `cv2.rectangle`, `cv2.putText` (label + motion label),
`cv2.polylines` (trajectory tối đa 20 điểm) — toàn bộ đều là lệnh Python gọi
OpenCV riêng lẻ, không gộp batch. Heatmap overlay dùng `cv2.addWeighted` toàn
frame khi `show_heatmap=true` (mặc định `true` trong `pipeline-live.yaml`).

**Ảnh hưởng:** Với ít người thì không đáng kể, nhưng đây là phần **luôn chạy
trên CPU** dù model đã infer trên GPU — với đông người, thời gian "drawing"
(`timing_ms.drawing`) có thể cạnh tranh đáng kể với thời gian model.

**Cải thiện có thể:**
- Tắt `show_hud`/`show_trajectories` khi không cần debug visual (đã tắt
  `show_hud` trong config live, tốt).
- Vẽ heatmap ở độ phân giải grid gốc rồi resize 1 lần (đang làm đúng), nhưng
  có thể bỏ qua hoàn toàn việc addWeighted khi `total_weight` quá nhỏ (đã có
  guard `> 0.0`, có thể nâng ngưỡng để tránh vẽ khi ảnh hưởng thị giác không
  đáng kể).

## 8. Luồng xử lý video upload (Gradio) tải lại toàn bộ model cho mỗi clip

**Vị trí:** `app.py::process_video_clip` → `_new_pipeline()`

Mỗi lần người dùng upload một video để phân tích, `_new_pipeline()` được gọi,
tạo **hoàn toàn mới** một `CrowdGenderPipeline` — load lại YOLO checkpoint,
YuNet, 2 classifier từ đĩa, chạy `warmup()` (bao gồm cả forward pass CUDA khởi
tạo kernel). Đây là chi phí vài trăm ms tới vài giây **trước khi** frame đầu
tiên của video được xử lý.

**Ảnh hưởng:** Với video ngắn, riêng chi phí load+warmup model có thể chiếm
tỷ lệ đáng kể trong tổng thời gian xử lý — đặc biệt tệ nếu người dùng thử
nhiều clip liên tiếp.

**Cải thiện có thể:**
- Giữ một pool/pipeline đã warm sẵn cho luồng "upload video" tương tự cách
  `LIVE_PIPELINES` đang làm cho webcam, chỉ gọi `pipeline.reset()` giữa các
  clip thay vì tạo instance mới hoàn toàn (cẩn trọng vì `reset()` không xoá
  model đã load, chỉ xoá tracker/analytics state — hoàn toàn phù hợp cho mục
  đích này).

## 9. FFmpeg re-encode 2 lần cho mỗi video upload

**Vị trí:** `app.py::process_video_clip`

Pipeline hiện tại: `ffmpeg normalize input -> OpenCV decode/process ->
OpenCV encode mp4v -> ffmpeg re-encode ra H.264 browser-compatible`. Tức là
**2 lần gọi FFmpeg subprocess** (normalize + final encode) cộng với việc dùng
codec `mp4v` trung gian (không hiệu quả) cho bước ghi frame trung gian.

**Ảnh hưởng:** Với video dài/lớn, đây là overhead I/O + CPU encode đáng kể,
chạy hoàn toàn tuần tự và đồng bộ (block toàn bộ request).

**Cải thiện có thể:**
- Ghi trực tiếp ra H.264 (`libx264`) ở bước OpenCV `VideoWriter` nếu backend
  hỗ trợ, bỏ bước re-encode thứ hai.
- Chạy bước "normalize" chỉ khi input codec thực sự không được OpenCV hỗ trợ
  (kiểm tra trước khi luôn luôn transcode).

## 10. Giới hạn concurrency ở tầng API/deployment (theo thiết kế, nhưng vẫn là bottleneck khi scale)

**Vị trí:** `src/api/sessions.py::DemoSessionManager` (`max_sessions` mặc định
1), `src/api/app.py::analyze_short_video` (chặn hoàn toàn khi có live session
hoạt động), `deploy/modal_app.py`

Hệ thống được thiết kế có chủ đích cho **1 GPU / 1 session** cho demo — đây là
quyết định kiến trúc hợp lý cho MVP, nhưng nó đồng nghĩa:
- Không có khả năng chạy nhiều camera đồng thời trên một instance.
- Yêu cầu phân tích video ngắn bị từ chối (`429`) hoàn toàn khi có 1 live
  session đang chạy, dù GPU có thể còn dư tài nguyên.
- Không thấy cấu hình multi-worker Uvicorn hoặc hàng đợi job — mỗi API
  instance tương đương một pipeline vật lý.

**Cải thiện có thể (nếu mục tiêu vượt ra khỏi phạm vi demo):**
- Thêm hàng đợi job (queue) cho short-video analysis độc lập với live session,
  cho phép chạy đồng thời nếu VRAM còn dư (thay vì chặn cứng theo
  `active_sessions > 0`).
- Horizontal scale bằng nhiều container/instance đứng sau load balancer khi
  cần nhiều camera, thay vì cố gắng scale trong 1 process.

## Tổng kết theo mức độ ảnh hưởng (ước lượng)

Mức ảnh hưởng thực tế phụ thuộc vào số người trong khung hình và phần cứng cụ
thể; thứ tự dưới đây dựa trên số lượng công việc mỗi bottleneck tạo ra khi
đông người / chạy liên tục:

1. Trích xuất face/body theo vòng lặp CPU tuần tự (mục 2) — tăng tuyến tính
   theo số track được lên lịch mỗi frame.
2. `PersonIdentityResolver.update` gần bậc hai theo số người (mục 6) — chỉ rõ
   rệt khi đông người thật.
3. Đồng bộ CPU↔GPU lặp lại (mục 3) — cố định overhead mỗi frame, ảnh hưởng
   mạnh khi frame rate cao hoặc batch nhỏ.
4. Serialize heatmap/zone 2 lần mỗi frame (mục 5) — overhead Python cố định,
   không phụ thuộc GPU.
5. Vẽ overlay per-track (mục 7) — nhỏ với vài người, tăng dần theo crowd size.
6. Reload model mỗi video upload + FFmpeg 2 lần (mục 8, 9) — one-off cost mỗi
   request, không ảnh hưởng throughput ổn định nhưng ảnh hưởng latency cảm
   nhận của người dùng.
7. Giới hạn concurrency ở tầng API (mục 10) — không phải "chậm" mà là giới
   hạn về khả năng mở rộng (scalability), cần cải thiện nếu đi ra khỏi phạm vi
   demo một-GPU-một-session.
8. Cấu hình detector ưu tiên recall (mục 4) — chi phí có chủ đích, chỉ nên
   điều chỉnh sau khi có dữ liệu telemetry thực tế từ `detector_statistics()`.
9. Kiến trúc single-stream-per-pipeline (mục 1) — giới hạn nền tảng, khó sửa
   nhanh vì gắn với cách Ultralytics quản lý tracker state; là điểm nghẽn
   "chiến lược" hơn là "chiến thuật".
