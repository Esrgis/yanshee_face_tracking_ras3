#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import json
import time
import sys
from core.vision_haarcascade import VisionHaarCascade
from core.control import PIDController

def load_config(path="config.json"):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    print("="*60)
    print(" FACE TRACKING - HAAR + KCF TRACKER + PID")
    print("="*60)

    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    pid_cfg = cfg.get("controller_pid", {})
    env_cfg = cfg.get("testing_env", {})

    # Setup camera
    w = cam_cfg.get("frame_width", 640)
    h = cam_cfg.get("frame_height", 480)
    cap = cv2.VideoCapture(env_cfg.get("webcam_index", 0))
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    center_x = w // 2

    # Init video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('output_tracking.avi', fourcc, 20.0, (w, h))
    print("[INFO] Recording video to output_tracking.avi")

    # Init vision
    vision = VisionHaarCascade(
        cascade_path=cam_cfg.get("cascade_path"),
        detection_skip=1  # detect every frame
    )

    # Init PID
    pid = PIDController(
        Kp=pid_cfg.get("Kp", 0.05),
        Ki=pid_cfg.get("Ki", 0.0),
        Kd=pid_cfg.get("Kd", 0.01),
        deadzone=pid_cfg.get("deadzone", 2.0),
        output_limit=pid_cfg.get("output_limit", 15.0)
    )

    frame_count = 0
    prev_time = time.time()

    print("\n[READY] Press 'q' to quit or Ctrl+C to stop and save video\n")

    try:
        while True:
            dt = time.time() - prev_time
            prev_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (w, h))
            frame_count += 1

            # Detect
            found, bbox, cx, cy = vision.process_frame(frame)

            error = 0.0
            output = 0.0
            if found and cx >= 0:
                error = float(cx - center_x)
                output = pid.update(error, dt if dt > 0 else 0.033)

            # Draw
            disp = frame.copy()
            if found and bbox:
                x, y, wb, hb = bbox
                cv2.rectangle(disp, (x,y), (x+wb, y+hb), (0,255,0), 2)
                cv2.circle(disp, (cx, cy), 5, (0,255,255), -1)
            cv2.line(disp, (center_x,0), (center_x,h), (255,0,0), 1)
            cv2.putText(disp, "Error: {:.1f}px  PID: {:.2f}deg".format(error, output),
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Write frame
            out_video.write(disp)

            # Show
            cv2.imshow("Face Tracking", disp)
            if frame_count % 10 == 0:
                fps = 1.0/dt if dt>0 else 0
                print("[Frame {}] found={} | error={:6.1f}px | output={:6.2f}deg | fps={:5.1f}".format(
                    frame_count, found, error, output, fps))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed, saving video...")
    finally:
        cap.release()      
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()