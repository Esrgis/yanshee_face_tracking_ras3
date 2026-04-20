#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_tracker.py -- Demo/debug tool, chay tren laptop KHONG can robot.
Muc dich: test vision pipeline doc lap (Haar + KCF).
Khong co FSM, scheduler, yanshee_interface.
"""
from __future__ import print_function
import cv2
import json
import time
import os
from core.vision_haarcascade import VisionHaarCascade
from core.control import PIDController
from core.filters import TrackerKalmanFilter
kalman   = TrackerKalmanFilter()
k_inited = False


def load_config(path="config.json"):
    base = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(base, path)
    if not os.path.exists(full):
        full = path
    with open(full, 'r') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print(" FACE TRACKING - DEMO (Laptop only, no robot)")
    print("=" * 60)

    cfg     = load_config()
    cam_cfg = cfg.get("camera", {})
    pid_cfg = cfg.get("controller_pid", {})
    env_cfg = cfg.get("testing_env", {})

    W = cam_cfg.get("frame_width",  640)
    H = cam_cfg.get("frame_height", 480)

    cap = cv2.VideoCapture(env_cfg.get("webcam_index", 0))
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    # fix: dung so thay constant name (tuong thich Pi + moi OpenCV build)
    cap.set(3,  W)
    cap.set(4,  H)
    cap.set(38, 1)  # MJPEG

    center_x = W // 2

    # Video writer
    fourcc    = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('output_tracking.avi', fourcc, 20.0, (W, H))
    print("[INFO] Recording -> output_tracking.avi")

    # Vision -- dung detection_skip tu config, khong hardcode
    vision = VisionHaarCascade(
        cascade_path    = cam_cfg.get("cascade_path"),
        detection_skip  = cam_cfg.get("detection_skip_frames", 1),
        pad_ratio       = cam_cfg.get("pad_ratio", 0.20),
        iou_reinit_threshold = cam_cfg.get("iou_reinit_threshold", 0.5),
        max_jump_px     = cam_cfg.get("max_jump_px", 180),
    )

    # PID
    pid = PIDController(
        Kp           = pid_cfg.get("Kp", 0.05),
        Ki           = pid_cfg.get("Ki", 0.0),
        Kd           = pid_cfg.get("Kd", 0.01),
        deadzone     = pid_cfg.get("deadzone", 2.0),
        output_limit = pid_cfg.get("output_limit", 15.0),
    )

    frame_count = 0
    prev_time   = time.time()

    print("[READY] Press 'q' to quit\n")

    try:
        while True:
            now   = time.time()
            dt    = now - prev_time
            prev_time = now

            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.resize(frame, (W, H))
            frame_count += 1

            found, bbox, cx, cy = vision.process_frame(frame)

            error  = 0.0
            output = 0.0
           # 3. Trong main loop — thay cx thẳng vào PID
            if found and cx >= 0:
                if not k_inited:
                    kalman.init_state(cx, cy)
                    k_inited = True
                    cx_f = cx
                else:
                    cx_f, _ = kalman.update(cx, cy)
                error  = float(cx_f - center_x)
                output = pid.update(error, dt if dt > 0 else 0.033)
            else:
                if k_inited:
                    cx_f, _ = kalman.predict()
                k_inited = False if not found else k_inited
                pid.reset_memory()
                error = output = 0.0
            # --- Draw ---
            disp = frame.copy()
            if found and bbox:
                x, y, wb, hb = bbox
                cv2.rectangle(disp, (x, y), (x + wb, y + hb), (0, 255, 0), 2)
                cv2.circle(disp, (cx, cy), 5, (0, 255, 255), -1)

            cv2.line(disp, (center_x, 0), (center_x, H), (255, 0, 0), 1)
            cv2.putText(
                disp,
                "Error: {:.1f}px  PID: {:.2f}  skip={}".format(
                    error, output, vision.detection_skip),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
            cv2.putText(
                disp,
                "found={}".format(found),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if found else (0, 0, 255), 2
            )

            out_video.write(disp)
            cv2.imshow("Face Tracking [DEMO]", disp)

            if frame_count % 30 == 0:
                fps = 1.0 / dt if dt > 0 else 0.0
                print("[f={:4d}] found={} | error={:6.1f}px | "
                      "pid={:6.2f} | fps={:5.1f} | skip={}".format(
                    frame_count, found, error, output, fps,
                    vision.detection_skip))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        print("[INFO] Video saved.")


if __name__ == "__main__":
    main()