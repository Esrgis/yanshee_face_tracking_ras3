# Yanshee Face Tracking System

Hệ thống theo dõi khuôn mặt sử dụng Haar Cascade và KCF tracker cho robot Yanshee.

## Tính năng chính

- Phát hiện khuôn mặt bằng Haar Cascade Classifier
- Theo dõi ổn định bằng KCF tracker
- Điều khiển PID cho robot
- Bộ lọc Kalman để làm mượt
- Máy trạng thái để quản lý hành vi

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy thử nghiệm

```bash
# Chạy với camera và hiển thị video
python main_tracker.py

# Chạy trên robot Yanshee (headless)
python main_tracker_robot.py
```

## Cấu hình

Chỉnh sửa file `config.json`:

```json
{
  "camera": {
    "cascade_path": "haarcascade_frontalface_default.xml",
    "frame_width": 640,
    "frame_height": 480
  },
  "controller_pid": {
    "Kp": 0.05,
    "Ki": 0.0,
    "Kd": 0.01
  },
  "robot_yanshee": {
    "max_angle": 180,
    "min_angle": 0,
    "default_angle": 90
  }
}
```

## Cấu trúc thư mục

```
├── main_tracker.py          # Script chính với UI
├── main_tracker_robot.py    # Script cho robot (headless)
├── config.json              # Cấu hình hệ thống
├── requirements.txt         # Dependencies
├── core/                    # Core modules
│   ├── vision.py           # Interface vision
│   ├── vision_haarcascade.py # Haar Cascade detector
│   ├── control.py          # PID controller
│   ├── filters.py          # Kalman filter
│   └── state_machine.py    # State machine
├── hardware/               # Hardware interfaces
│   ├── yanshee_interface.py # Robot interface
│   └── cam_stream.py       # Camera streaming
├── utils/                  # Utilities
└── data/                   # Data và logs
```

## Cách hoạt động

1. **Vision**: Phát hiện khuôn mặt bằng Haar Cascade, theo dõi bằng KCF
2. **Control**: Tính toán lỗi vị trí, điều khiển PID
3. **Filter**: Lọc nhiễu bằng Kalman filter
4. **State Machine**: Quản lý trạng thái (TRACKING, SEARCH, LOST, etc.)

## Test

```bash
# Test Haar Cascade detector
python test_haar.py
```

