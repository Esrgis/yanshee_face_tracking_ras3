# Yanshee Face Tracking: Adaptive Detection Scheduler

Hệ thống theo dõi khuôn mặt tối ưu hóa cho thiết bị nhúng tài nguyên thấp (Raspberry Pi 3) trên robot Yanshee. Dự án đề xuất và đánh giá một bộ định tuyến nhảy khung hình thích nghi (**Adaptive Detection Scheduler**), kết hợp với Haar Cascade và KCF Tracker.

## Đóng góp chính
- **Adaptive Detection Scheduler:** Tự động điều chỉnh tần suất quét khuôn mặt (detection skip) dựa trên vận tốc mục tiêu và độ rung nhiễu (jitter), giúp duy trì FPS cao trên cấu hình yếu.
- **Ablation Study Framework:** Tích hợp bộ công cụ tự động chạy các kịch bản so sánh, trích xuất metrics (Recovery time, Jitter, FPS) và vẽ biểu đồ phân tích.
- **Kiến trúc Pipeline:** Kết hợp Tracking-by-detection, Bộ lọc Kalman (mượt mà hóa quỹ đạo) và Bộ điều khiển PID (điều khiển Servo robot).

## Kiến trúc thư mục

\`\`\`text
├── core/
│   ├── vision_haarcascade.py  # Haar Cascade + KCF Tracker
│   ├── adaptive_sheduler.py   # Thuật toán Adaptive Frame Skipping
│   ├── filters.py             # Kalman Filter
│   ├── control.py             # PID Controller
│   └── state_machine.py       # FSM (SEARCH, TRACK, LOST)
├── hardware/
│   └── yanshee_interface.py   # Giao tiếp với API của robot Yanshee
├── scripts/
│   ├── run_ablation.py        # Chạy 4 config so sánh hiệu năng
│   └── analyze_results.py     # Đọc CSV và vẽ biểu đồ Matplotlib
├── results/
│   ├── logs/                  # Chứa file CSV xuất ra từ Ablation
│   └── figures/               # Chứa biểu đồ kết quả (fig1..fig4)
├── Makefile                   # Cấu hình lệnh chạy nhanh
├── config.json                # Cấu hình tham số hệ thống
└── main_tracker_robot.py      # Script chạy thực tế trên Robot
\`\`\`

## Cài đặt

Dự án yêu cầu **Python 3.5.3** (Để tương thích ngược với Yanshee SDK):

\`\`\`bash
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường (Windows)
.\venv\Scripts\activate.ps1
# (Hoặc trên Linux/Raspberry Pi): source venv/bin/activate

# 3. Cài đặt thư viện
pip install -r requirements.txt
\`\`\`

## Hướng dẫn sử dụng (Qua Makefile)

Thay vì phải gõ lệnh dài dòng, dự án cung cấp các lệnh `make` tiện lợi:

### 1. Thu thập dữ liệu nghiên cứu (Ablation Study)
Chạy tự động 4 cấu hình (Config A, B, C, D) để so sánh hiệu năng của thuật toán.
\`\`\`bash
make ablation
# Tùy chỉnh tham số: make ablation SRC=video.mp4 DURATION=60 CONFIGS=ABCD
\`\`\`

### 2. Phân tích kết quả
Đọc các file log CSV vừa sinh ra, tính toán các chỉ số (Recovery time, Mean Jitter, Tracking Rate) và xuất ra biểu đồ png.
\`\`\`bash
make analyze
\`\`\`
*(Kết quả sẽ nằm trong thư mục `results/figures/`)*

### 3. Deploy lên Robot Yanshee
Chạy hệ thống nhận diện và điều khiển motor trong thực tế (Headless mode).
\`\`\`bash
make robot
\`\`\`

## Các cấu hình Ablation (Ablation Configs)
* **A_static_skip1:** Quét khuôn mặt ở mọi frame (Baseline nặng nhất).
* **B_static_skip5:** Cố định bỏ qua 5 frame (Baseline nhẹ).
* **C_adaptive_vel_only:** Nhảy frame thích nghi dựa trên vận tốc mục tiêu (Beta = 0).
* **D_adaptive_full:** Nhảy frame thích nghi dựa trên cả vận tốc và độ rung nhiễu (Proposed Method).