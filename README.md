
# Yanshee Visual Tracking System

## Overview
This repository contains the core software stack for an intelligent, real-time visual tracking system designed for the Yanshee Robot (and adaptable to general Pan-Tilt robotic platforms). The system addresses non-linear mechanical boundary constraints and visual jitter by integrating Deep Learning detection, Kalman Filtering, and a Finite State Machine (FSM).

## System Architecture
The pipeline is designed with strict Separation of Concerns, isolating the algorithmic logic from the Hardware Abstraction Layer:
* **Vision System (`core/vision_*.py`):** Multi-backend support (YOLOv8/v12 via ONNX or PyTorch) combined with a KCF Tracker (Cascade Tracking) to maintain high frame rates on edge devices.
* **Signal Filter (`core/filters.py`):** Utilizes a Kalman Filter to predict object trajectories during frame drops and to smooth high-frequency spatial jitter.
* **Controller (`core/control.py`):** A custom PID controller decoupled from frame rates (utilizing dynamic `dt` calculation), featuring an Active Anti-Windup mechanism.
* **State Machine (`core/state_machine.py`):** Manages four physical states (`TRACKING`, `SATURATED`, `SEARCH`, `LOST`) to govern hardware safety and prevent integral windup when servos reach mechanical limits.

## Repository Structure

```text
yanshee_visual_tracking/
├── core/                       # Core algorithm modules
│   ├── __init__.py
│   ├── vision.py               # Vision System Interface (ABC)
│   ├── vision_onnx.py          # Vision backend using ONNX (Optimized for CPU)
│   ├── vision_ultralytics.py   # Vision backend using PyTorch (.pt)
│   ├── control.py              # PID Controller implementation
│   ├── filters.py              # Kalman & Moving Average Filters
│   └── state_machine.py        # Tracking logic FSM
├── hardware/                   # Hardware abstraction layer
│   ├── __init__.py
│   └── yanshee_interface.py    # Yanshee servo/SDK communication class
├── models/                     # AI weights directory (.pt, .onnx)
├── utils/                      # Evaluation and conversion utilities 
│   ├── __init__.py
│   ├── converter.py            # Converts ground truth data formats
│   └── evaluation.py           # Calculates metrics (IoU, RMSE, Stability)
├── data/                       # Mockup testing data and logs
├── config.json                 # Centralized configuration file
├── main_tracker.py             # Main entry point script
└── requirements.txt            # Environment dependencies
```

## Environment Setup

**Note on dependencies:** The system relies on the legacy OpenCV KCF Tracker. To avoid library conflicts, ensure that `opencv-python` and `opencv-contrib-python` are not installed simultaneously in the same environment.

**Step 1:** Install Python 3.11.

**Step 2:** Create and activate a virtual environment.

On Linux/macOS:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python3.11 -m venv venv
venv\Scripts\activate
```

**Step 3:** Install required packages:
```bash
pip install -r requirements.txt
```
*(Key dependencies include: `opencv-contrib-python`, `ultralytics`, `onnx`, `onnxruntime`, and `numpy`).*

## Configuration & Usage

The system's operational parameters are centralized within `config.json`.

### 1. Model Backend Selection
Locate the `"camera"` block in `config.json` to switch the active model backend:

* **For ONNX (Recommended for CPU/Edge Devices):**
  ```json
  "camera": {
      "model_path": "models/yolov8n-face-lindevs.onnx",
      "model_backend": "onnx",
      ...
  }
  ```
* **For PyTorch (.pt):**
  ```json
  "camera": {
      "model_path": "models/yolov12n-face.pt",
      "model_backend": "pt",
      ...
  }
  ```

### 2. Video Source Configuration
Locate the `"testing_env"` block in `config.json`:
* **Live Camera Interface:** Set `"use_live_camera": true`.
* **Video File (Offline Mocking):** Set `"use_live_camera": false` and specify the path in `"video_source"` (e.g., `"data/videos/test_sequence.mp4"`).

### 3. Execution
Run the main tracking script from the root directory:
```bash
python main_tracker.py
```
*(Press `q` within the video window to terminate the process gracefully and save metric logs).*
```

