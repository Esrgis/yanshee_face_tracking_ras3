# Yanshee Robot - Intelligent Visual Tracking System

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.9.0-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange.svg)

## 📌 Overview
This repository contains the core software stack for an intelligent, real-time visual tracking system designed for the Yanshee Robot (or any Pan-Tilt robotic head). The system solves the non-linear mechanical boundary problem and vision jitter by combining Deep Learning, Kalman Filtering, and a Finite State Machine (FSM).

## 🧠 System Architecture
The pipeline is designed with strict **Separation of Concerns** (Hardware Abstraction Layer):
1. **Vision System (`core/vision.py`):** YOLOv8/v12 (Detection) + KCF Tracker (Cascade Tracking) to maintain high FPS on edge devices.
2. **Signal Filter (`core/filters.py`):** Kalman Filter predicts object trajectory during frame drops and smooths high-frequency camera jitter.
3. **Controller (`core/control.py`):** A custom PID controller strictly decoupled from frame rates (calculates real `dt`) with an **Active Anti-Windup** mechanism.
4. **State Machine (`core/state_machine.py`):** Manages 4 physical states (`TRACKING`, `SATURATED`, `SEARCH`, `LOST`) to prevent integral windup when the servo hits its mechanical boundaries.

## 📂 Project Structure
```text
yanshee_visual_tracking/
├── core/                   # Core algorithms (Vision, PID, Kalman, FSM)
├── hardware/               # Hardware Abstraction Layer (Yanshee Mock & Serial)
├── data/                   # Videos for offline testing & CSV logs
├── config.json             # Centralized configuration (PID tuning, boundaries)
├── main_tracker.py         # Main entry point and execution loop
├── requirements.txt        # Pinned dependencies
└── README.md


# Installation & Setup

git clone [https://github.com/your-username/yanshee_visual_tracking.git](https://github.com/your-username/yanshee_visual_tracking.git)
cd yanshee_visual_tracking

## Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install Dependencies:

pip install -r requirements.txt
