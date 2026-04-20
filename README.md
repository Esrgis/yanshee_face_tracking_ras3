# Yanshee Face Tracking: Adaptive Detection Scheduler

> Hệ thống visual tracking thời gian thực cho robot Yanshee (Raspberry Pi 3).  
> Không phải benchmark Computer Vision thông thường —  
> đây là bài toán **sinh tồn của hệ thống điều khiển vòng kín trên edge device**.

---

## Tại sao dự án này tồn tại

Raspberry Pi 3 quá yếu để chạy face detection mỗi frame. Nếu detect liên tục → FPS xuống còn 3–5 → robot lag 300ms+ → điều khiển vô nghĩa.

**Giải pháp cốt lõi:** Tách detection (chạy thưa) khỏi tracking (chạy mọi frame), và tự động điều chỉnh tần suất detection theo hành vi của target.

---

## Pipeline hệ thống

```text
Camera Frame
↓
[Adaptive Scheduler]  ← quyết định có chạy detection frame này không
↓
[Detector: Haar/LBP/SSD]  ← tìm mặt (chạy thưa)
↓
[KCF/MOSSE Tracker]  ← giữ mặt giữa các lần detect (chạy mọi frame)
↓
[Linear Kalman Filter]  ← làm mượt tọa độ, lọc nhiễu
↓
[PID Controller]  ← tính góc quay servo
↓
[Robot Yanshee]  ← thực thi
```

**State Machine (FSM)** chạy song song, quản lý 4 trạng thái:

| State | Ý nghĩa | Scheduler |
|-------|---------|-----------|
| `TRACKING` | Đang theo dõi ổn định | Adaptive formula |
| `SATURATED` | Servo đã quay hết cỡ | Force skip=1 |
| `SEARCH` | Mất target tạm thời | Force skip=1 |
| `LOST` | Mất hoàn toàn | Force skip=1 |

---

## Adaptive Scheduler — đóng góp chính

**Công thức:**

`skip(t) = clip( base_skip / (1 + α·velocity + β·jitter), min_skip, max_skip )`

| Tham số | Ý nghĩa |
|---------|---------|
| `α = 0.08` | Target di chuyển nhanh → detect thường hơn |
| `β = 0.05` | Tracker bất ổn (jitter cao) → re-anchor sớm hơn |
| `base_skip = 5` | Điểm xuất phát khi target đứng yên |

Khi FSM ở trạng thái SEARCH/LOST/SATURATED → scheduler override cứng về `skip=1` (detect mọi frame để tìm lại target).

---

## Cấu trúc thư mục

```text
├── core/                        ← BLACK BOX. Không sửa.
│   ├── vision.py                # Abstract interface
│   ├── vision_haarcascade.py    # Haar Cascade + KCF Tracker
│   ├── vision_lbp.py            # LBP Cascade + KCF Tracker
│   ├── vision_ssd.py            # SSD MobileNet + KCF Tracker
│   ├── adaptive_scheduler.py    # Adaptive Detection Scheduler
│   ├── filters.py               # Linear Kalman Filter
│   ├── control.py               # PID Controller (Ki=0 mặc định → PD)
│   └── state_machine.py         # FSM: TRACKING/SATURATED/SEARCH/LOST
│
├── hardware/
│   ├── yanshee_interface.py     # Giao tiếp API robot Yanshee
│   ├── client_monitor.py        # Monitor client
│   └── server_record.py         # Record server
│
├── models/                      # Model files (không phải weights tracker)
│   ├── haarcascade_frontalface_default.xml
│   ├── lbpcascade_frontalface_improved.xml
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
│
├── scripts/
│   ├── run_benchmark.py         # Bước 1: so sánh 3 detector (Haar/LBP/SSD)
│   ├── run_ablation.py          # Bước 2: so sánh 5 config A→E
│   └── analyze_results.py       # Vẽ figures + xuất table_summary.csv
│
├── annotation/                  # Chạy trên laptop Python 3.10, KHÔNG lên Pi
│   ├── generate_pseudo_labels.py# YOLOv8 → draft annotation
│   ├── review_labels.py         # Review + sửa tay
│   └── requirements_annotation.txt
│
├── data/
│   ├── videos/                  # 4 clip: slow, normal, fast, scale
│   └── annotations/             # Ground truth JSON sau khi review
│       ├── slow.json
│       ├── normal.json
│       ├── fast.json
│       └── scale.json
│
├── results/
│   ├── logs/                    # CSV output từ benchmark + ablation
│   └── figures/                 # PNG figures + table_summary.csv
│
├── spec/
│   ├── benchmark_spec.yaml      # Định nghĩa metrics benchmark
│   └── ablation_spec.yaml       # Định nghĩa metrics ablation
│
├── main_tracker.py              # Demo/debug trên laptop (không cần robot)
├── main_tracker_robot.py        # Production: chạy thật trên Pi + Yanshee
├── config.json                  # Tham số hệ thống
├── requirements.txt             # Python 3.5 — dành cho Pi
└── Makefile                     # Lệnh tắt
```

---

## Cài đặt

### Trên Raspberry Pi / môi trường chính (Python 3.5)
```bash
python -m venv venv
source venv/bin/activate          # Linux/Pi
# .\venv\Scripts\activate.ps1     # Windows
pip install -r requirements.txt
```

### Trên laptop — chỉ để tạo Ground Truth annotation (Python 3.10)
```bash
pip install -r annotation/requirements_annotation.txt
```

> ⚠️ Không cài `requirements_annotation.txt` lên Pi — `ultralytics` không tương thích Python 3.5.

---

## Quy trình sử dụng

### Bước 0 — Tạo Ground Truth (laptop, 1 lần duy nhất)
```bash
python annotation/generate_pseudo_labels.py --video data/videos/slow.avi
python annotation/review_labels.py          # xem + sửa tay
# Output: data/annotations/slow.json → commit vào git
```

### Bước 1 — Benchmark detector
```bash
make benchmark
# So sánh Haar / LBP / SSD trên 4 video clip
# Output: results/logs/benchmark_summary.csv
```

### Bước 2 — Ablation Study
```bash
make ablation
# Hoặc: make ablation SRC=data/videos/normal.avi DURATION=60 CONFIGS=ABCD
# So sánh 5 config A→E
# Output: results/logs/ablation_*.csv
```

### Bước 3 — Phân tích kết quả
```bash
make analyze
# Output: results/figures/*.png + results/figures/table_summary.csv
```

### Bước 4 — Chạy thật trên robot
```bash
make robot
```

### Debug nhanh trên laptop (không cần robot)
```bash
python main_tracker.py
```

---

## 5 cấu hình Ablation

| Config | Tracker | Skip | α | β | Mô tả |
|--------|---------|------|---|---|-------|
| A | KCF | 1 (fixed) | — | — | Baseline nặng nhất, detect mọi frame |
| B | KCF | 5 (fixed) | — | — | Baseline nhẹ, skip cố định |
| C | KCF | Adaptive | 0.08 | 0 | Adaptive theo velocity |
| D | KCF | Adaptive | 0.08 | 0.05 | **Proposed method** — velocity + jitter |
| E | MOSSE | 5 (fixed) | — | — | Đối chứng tracker: MOSSE vs KCF |

**Primary metric:** `tracking_rate` (% frame giữ được target)  
**Secondary:** `fps_avg` (phải ≥ 15 FPS trên Pi 3)  
**Tertiary:** `jitter_mean` (chỉ tính khi `found=1`, undefined khi lost)

---

## Định nghĩa metrics quan trọng

| Metric | Định nghĩa |
|--------|-----------|
| `tracking_rate` | frames_found / total_frames |
| `system_fps` | FPS của toàn vòng lặp (bao gồm IO, scheduler, inference, render) |
| `inference_fps_theoretical` | 1000 / inference_ms_mean (chỉ `process_frame()`) |
| `jitter` | `|cx_raw − cx_filtered|` — undefined (không phải 0) khi lost |
| `scheduled_detection_successes` | Số lần detector chạy VÀ tìm thấy mặt — observable scheduler outcome |
| `iou_mean` | Tính trên toàn bộ GT frames; FN đóng góp 0 (detection-aware) |

---

## Cho người viết paper

Claim chính của paper:

> *"Adaptive Detection Scheduler with FSM-aware override achieves Pareto improvement on tracking_rate × fps trade-off compared to static skip baselines on Raspberry Pi 3."*

Novelty: chưa có paper nào apply velocity+jitter adaptive scheduling cho humanoid robot + PID servo với FSM-aware override.

Figures cho paper:
- **Fig 1:** Scatter `fps_avg` vs `tracking_rate` (A→E) — Pareto frontier
- **Fig 2:** Boxplot `jitter_mean` theo config
- **Fig 3:** Line `cx_filtered` overlay A, D, E — visualize stability
- **Fig 4:** Bar `sched_skip_mean` (C, D only)
- **Fig 5:** Bar `precision/recall/f1/iou` theo detector (từ benchmark)

Spec đầy đủ: xem `spec/benchmark_spec.yaml` và `spec/ablation_spec.yaml`.

---

## Lưu ý kỹ thuật

- `core/` là black box — **không sửa** dù bất kỳ lý do gì
- Tracker (KCF/MOSSE) không cần file weights — thuật toán toán học thuần túy trong `opencv-contrib`
- MOSSE trên opencv ≥ 4.5.1 dùng `cv2.legacy.TrackerMOSSE_create()` thay vì `cv2.TrackerMOSSE_create()`
- `config.json` chỉ dành cho production (robot thật) — không chứa ablation config
- Ground truth annotation workflow chỉ chạy trên laptop Python 3.10, output JSON commit vào git, Pi chỉ đọc JSON