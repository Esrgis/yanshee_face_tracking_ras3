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