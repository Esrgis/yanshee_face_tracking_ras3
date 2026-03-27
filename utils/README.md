# Converter (Label Conversion)

- **File:** [utils/converter.py](utils/converter.py) — công cụ chuyển nhãn YOLO/COCO sang file bbox per-image.
- **Mục đích:** Chuẩn hoá ground-truth sang dạng dễ dùng cho evaluation hoặc cho các module Vision. Hỗ trợ hai định dạng đầu ra: `minimal` và `full`.
- **Định dạng đầu ra:**
  - **minimal:** mỗi dòng `x y w h` (pixel, top-left, integer) — tương thích trực tiếp với `vision_onnx.py` / `vision_ultralytics.py`.
  - **full:** mỗi dòng `x y w h category_id score`.
- **Ví dụ chạy (YOLO labels):**

```bash
python -m yanshee_visual_tracking.utils.converter --input-type yolo --labels path/to/labels --images path/to/images --out path/to/out --out-format minimal
```

- **Ví dụ chạy (COCO json):**

```bash
python -m yanshee_visual_tracking.utils.converter --input-type coco --coco annotations.json --out path/to/out --out-format full
```

- **Yêu cầu:** Python + OpenCV (dùng để đọc kích thước ảnh). Nếu không có OpenCV, cung cấp kích thước ảnh bằng cách khác.
- **Kiểm tra:** Converter có validator cơ bản — nó kiểm tra cấu trúc file YOLO/COCO trước khi chuyển.
- **Ghi chú tích hợp:** Để dùng kết quả với `vision_onnx.py`, chọn `--out-format minimal` và đặt các file `.txt` per-image vào thư mục labels/ tương ứng khi cần.
# Data Collector (Thu Thập Dữ Liệu Camera)

- **File:** [utils/data_collector.py](data_collector.py) — công cụ thu thập dữ liệu video từ camera với giao diện real-time.
- **Mục đích:** Ghi lại các video chuyển động khác nhau (chậm, nhanh, đột ngột) cho việc training hoặc kiểm thử tracking.
- **Cấu hình:**
  - **Độ phân giải:** 640 × 480 pixels
  - **Tốc độ:** 25 FPS
  - **Định dạng output:** AVI (codec MJPEG)
  - **Thư mục lưu:** `data/videos/`

- **Phím tắt:**
  - **1:** Bắt đầu ghi video chuyển động chậm (Slow Motion)
  - **2:** Bắt đầu ghi video chuyển động nhanh (Fast Motion)
  - **3:** Bắt đầu ghi video chuyển động đột ngột (Sudden Motion)
  - **S:** Dừng ghi video
  - **Q:** Thoát chương trình

- **Cách chạy:**

```bash
cd /path/to/yanshee_visual_tracking
python -m utils.data_collector
```

Hoặc:

```bash
python utils/data_collector.py
```

- **Giao diện hiển thị:**
  - Số frame đã thu thập hiện tại
  - Trạng thái ghi video (loại + số frame đã ghi)
  - Hướng dẫn phím tắt

- **Tên file output:**
  - `slow_motion_YYYYmmdd_HHMMSS.avi` (video chuyển động chậm)
  - `fast_motion_YYYYmmdd_HHMMSS.avi` (video chuyển động nhanh)
  - `sudden_motion_YYYYmmdd_HHMMSS.avi` (video chuyển động đột ngột)

- **Yêu cầu:** Python + OpenCV (`pip install opencv-python`).
- **Lưu ý:** Đảm bảo camera/webcam đã được kết nối và sẵn sàng trước khi chạy chương trình. 
# Face Labeling Tool (Đánh nhãn Khuôn mặt)

- **File:** [utils/face_labeling_tool.py](face_labeling_tool.py) — công cụ phát hiện và đánh nhãn khuôn mặt từ camera hoặc video, lưu annotations theo định dạng YOLO1.1 cho CVAT.
- **Mục đích:** Tự động phát hiện khuôn mặt trong video và lưu lại annotations có thể import vào CVAT để tinh chỉnh hoặc kiểm tra.
- **Model:** YOLOv8n-face-lindevs (`models/yolov8n-face-lindevs.pt`)
- **Cấu hình:**
  - **Ngưỡng Confidence:** 0.5 (mặc định, có thể tùy chỉnh)
  - **Định dạng output:** YOLO1.1 (normalized: `class_id center_x center_y width height`)
  - **Thư mục lưu nhãn:** `data/labels_yolo_[video_name]/`

- **Cách chạy:**

1. **Camera (mặc định):**
   ```bash
   python utils/face_labeling_tool.py
   ```

2. **Video cụ thể:**
   ```bash
   python utils/face_labeling_tool.py data/videos/slow_motion_20260328_052704.avi
   ```

3. **Tất cả videos trong `data/videos/` (Khuyên dùng):**
   ```bash
   python utils/face_labeling_tool.py --all
   ```

- **Phím tắt:**
  - **q:** Thoát
  - **p:** Pause/Resume (khi xử lý video)

- **Output:**
  - Mỗi frame được lưu với file label `.txt` tương ứng
  - File `metadata.json` chứa thông tin video (fps, resolution, tổng frames)
  - Ví dụ cấu trúc:
    ```
    data/labels_yolo_slow_motion_*/
    ├── metadata.json
    ├── slow_motion_000001.txt
    ├── slow_motion_000002.txt
    └── ...
    ```

- **Định dạng YOLO1.1:**
  ```
  class_id center_x center_y width height
  0 0.523456 0.456789 0.234567 0.345678
  ```
  - Tất cả giá trị được normalize (0-1)
  - class_id = 0 cho face

- **Giao diện hiển thị:**
  - Bounding box xung quanh khuôn mặt (xanh lá)
  - Confidence score cho mỗi detection
  - Số lượng khuôn mặt phát hiện được
  - Tiến trình xử lý (Frame/Total)
  - Thông tin folder lưu labels

- **Yêu cầu:** Python + OpenCV + Ultralytics YOLO (`pip install ultralytics opencv-python`).
- **Lưu ý:** Annotations được lưu tự động, có thể import vào CVAT để tinh chỉnh hoặc kiểm tra chất lượng. Đảm bảo model `.pt` tồn tại trong thư mục `models/` trước khi chạy.
