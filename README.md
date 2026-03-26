# Yanshee Visual Tracking System 🤖👁️

## 📌 Overview
This repository contains the core software stack for an intelligent, real-time visual tracking system designed for the Yanshee Robot (or any Pan-Tilt robotic head). The system solves the non-linear mechanical boundary problem and vision jitter by combining Deep Learning, Kalman Filtering, and a Finite State Machine (FSM).

## 🧠 System Architecture
The pipeline is designed with strict **Separation of Concerns** (Hardware Abstraction Layer):
* **Vision System (`core/vision.py`, `vision_onnx.py`, `vision_ultralytics.py`):** Multi-backend support (YOLOv8/v12 via ONNX or PyTorch) + KCF Tracker (Cascade Tracking) to maintain high FPS on edge devices.
* **Signal Filter (`core/filters.py`):** Kalman Filter predicts object trajectory during frame drops and smooths high-frequency camera jitter.
* **Controller (`core/control.py`):** A custom PID controller strictly decoupled from frame rates (calculates real `dt`) with an Active Anti-Windup mechanism.
* **State Machine (`core/state_machine.py`):** Manages 4 physical states (`TRACKING`, `SATURATED`, `SEARCH`, `LOST`) to prevent integral windup when the servo hits its mechanical boundaries.

## 📂 Folder Structure

```text
yanshee_visual_tracking/
├── core/                       # Core algorithm modules
│   ├── __init__.py
│   ├── vision.py               # Vision System Interface (ABC)
│   ├── vision_onnx.py          # Vision backend using ONNX (Optimized for CPU)
│   ├── vision_ultralytics.py   # Vision backend using PyTorch (.pt)
│   ├── control.py              # PID Controller
│   ├── filters.py              # Kalman & Moving Average Filters
│   └── state_machine.py        # FSM (State Machine)
├── hardware/                   # Hardware communication layer
│   ├── __init__.py
│   └── yanshee_interface.py    # Yanshee servo/SDK control class
├── models/                     # AI weights directory (.pt, .onnx)
├── utils/                      # Utilities 
│   ├── __init__.py
|   ├── converter.py            # Convert file groundtruth to 
│   └── evaluation.py           # 
├── data/                       # Mockup testing data (videos)
├── config.json                 # Centralized configuration file
├── main_tracker.py             # Main entry point script
└── requirements.txt            # Environment dependencies
```

## ⚙️ Environment Setup

**⚠️ CRITICAL WARNING:** The system relies on the KCF Tracker (a legacy OpenCV tracker). To avoid library conflicts, you **MUST NOT** install both `opencv-python` and `opencv-contrib-python` at the same time.

**Step 1:** Install python3.11

**Step 2:** Install required dependencies:
```bash
pip install -r requirements.txt
```
*(Key libraries: `opencv-contrib-python`, `ultralytics`, `onnx`, `onnxruntime`, `numpy`)*

## 🚀 Configuration & Usage

The system's behavior is fully controlled via `config.json`.

### 1. Switch AI Backend (Model)
Open `config.json`, locate the `"camera"` block. You can easily hot-swap the model backend by changing `"model_backend"` and `"model_path"`:

* **Use ONNX (Recommended for CPU/Edge Devices):**
  ```json
  "camera": {
      "model_path": "models/yolov8n-face-lindevs.onnx",
      "model_backend": "onnx",
      ...
  }
  ```
* **Use Ultralytics (.pt):**
  ```json
  "camera": {
      "model_path": "models/yolov12n-face.pt",
      "model_backend": "pt",
      ...
  }
  ```

### 2. Video Source Selection
In `config.json`, find the `"testing_env"` block:
* **Live Camera:** Set `"use_live_camera": true`
* **Video File (Mocking):** Set `"use_live_camera": false` and update `"video_source"` to your video path (e.g., `"data/videos/videoplayback.mp4"`).

### 3. Run the Tracker
Run the main script from the root directory:
```bash
python main_tracker.py
```
*(Press `q` on the video window to gracefully exit and save logs).*

