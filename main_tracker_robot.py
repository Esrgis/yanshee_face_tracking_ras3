#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_tracker_robot.py -- Production pipeline cho Yanshee robot.
Chay tren Pi (Python 3.5) hoac laptop (simulation mode).

Usage:
  python main_tracker_robot.py                  # sim mode
  python main_tracker_robot.py --real           # real robot
  python main_tracker_robot.py --config D       # chon scheduler config
"""
from __future__ import print_function
import cv2
import json
import time
import os
import sys
import argparse

from core.vision_haarcascade import VisionHaarCascade
from core.filters             import TrackerKalmanFilter
from core.control             import PIDController
from core.state_machine       import TrackingStateMachine, RobotState
from core.adaptive_scheduler  import AdaptiveDetectionScheduler
from hardware.yanshee_interface import YansheeInterface


def load_config(path="config.json"):
    base = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(base, path)
    if not os.path.exists(full):
        full = path
    with open(full, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",   action="store_true", help="Chay robot that (mac dinh: sim)")
    ap.add_argument("--config", default=None,        help="Override scheduler config (A/B/C/D)")
    ap.add_argument("--ui",     action="store_true", help="Hien thi cua so OpenCV")
    args = ap.parse_args()

    cfg         = load_config()
    vis_cfg     = cfg.get("vision", {})
    pid_cfg     = cfg.get("controller_pid", {})
    kal_cfg     = cfg.get("filter_kalman", {})
    rob_cfg     = cfg.get("robot_yanshee", {})
    hw_cfg      = cfg.get("hardware", {})
    sched_cfg   = cfg.get("adaptive_scheduler", {})
    sys_cfg     = cfg.get("system", {})
    env_cfg     = cfg.get("testing_env", {})

    is_sim      = not args.real
    show_ui     = args.ui or sys_cfg.get("display_ui", False)

    W = vis_cfg.get("frame_width",  320)
    H = vis_cfg.get("frame_height", 240)
    center_x = W // 2

    # --- Chon scheduler config ---
    config_key  = args.config or sched_cfg.get("active_config", "D")
    configs     = sched_cfg.get("configs", {})
    sel_cfg     = configs.get(config_key, configs.get("D", {}))

    print("=" * 60)
    print(" YANSHEE FACE TRACKER | mode={} | scheduler={}".format(
        "SIM" if is_sim else "REAL", config_key))
    print("=" * 60)

    # --- Init camera ---
    cap = cv2.VideoCapture(env_cfg.get("webcam_index", 0))
    cap.set(3, W)
    cap.set(4, H)
    cap.set(38, 1)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    # --- Init components ---
    vision = VisionHaarCascade(
        cascade_path         = vis_cfg.get("models", {}).get("haarcascade"),
        detection_skip       = sel_cfg.get("base_skip", 5),
        pad_ratio            = vis_cfg.get("pad_ratio", 0.20),
        iou_reinit_threshold = vis_cfg.get("iou_reinit_threshold", 0.5),
    )

    kalman = TrackerKalmanFilter(
        process_noise      = kal_cfg.get("process_noise_cov", 0.03),
        measurement_noise  = kal_cfg.get("measurement_noise_cov", 0.1),
    )
    k_inited = False

    pid = PIDController(
        Kp           = pid_cfg.get("Kp", 0.05),
        Ki           = pid_cfg.get("Ki", 0.0),
        Kd           = pid_cfg.get("Kd", 0.01),
        max_integral = pid_cfg.get("max_integral", 50.0),
        deadzone     = pid_cfg.get("deadzone", 2.0),
        output_limit = pid_cfg.get("output_limit", 15.0),
    )

    fsm = TrackingStateMachine(
        timeout_lost         = rob_cfg.get("timeout_lost", 3.0),
        lost_frame_threshold = rob_cfg.get("lost_frame_threshold", 5),
    )

    scheduler = AdaptiveDetectionScheduler(
        enabled         = sel_cfg.get("alpha", 0.0) > 0 or sel_cfg.get("beta", 0.0) > 0,
        base_skip       = sel_cfg.get("base_skip", 5),
        min_skip        = sched_cfg.get("min_skip", 1),
        max_skip        = sched_cfg.get("max_skip", 15),
        alpha           = sel_cfg.get("alpha", 0.0),
        beta            = sel_cfg.get("beta", 0.0),
        velocity_window = sched_cfg.get("velocity_window", 5),
    )

    robot = YansheeInterface(hw_cfg, is_simulation=is_sim)

    servo_center  = rob_cfg.get("servo_center",  90.0)
    servo_min_abs = rob_cfg.get("servo_min_abs", 15.0)
    servo_max_abs = rob_cfg.get("servo_max_abs", 165.0)

    # --- Main loop ---
    frame_count = 0
    prev_time   = time.time()
    cx_filtered = -1
    jitter      = 0.0

    print("[READY] Ctrl+C de dung\n")

    try:
        while True:
            now  = time.time()
            dt   = now - prev_time
            prev_time = now

            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.resize(frame, (W, H))
            frame_count += 1

            # 1. Scheduler → cap nhat skip
            state      = fsm.get_state()
            skip       = scheduler.compute_skip(cx_filtered, -1, jitter, state)
            vision.detection_skip = skip

            # 2. Vision
            found, bbox, cx_raw, cy_raw = vision.process_frame(frame)

            # 3. Kalman
            if found and cx_raw >= 0:
                if not k_inited:
                    kalman.init_state(cx_raw, cy_raw)
                    k_inited    = True
                    cx_filtered = cx_raw
                else:
                    cx_filtered, _ = kalman.update(cx_raw, cy_raw)
                jitter = abs(cx_raw - cx_filtered)
            else:
                if k_inited:
                    cx_filtered, _ = kalman.predict()
                jitter = 0.0

            # 4. FSM
            pid_output  = 0.0
            neck_abs    = servo_center
            state       = fsm.update(
                found,
                servo_center - pid_output,
                servo_max_abs,
                servo_min_abs,
            )

            # 5. PID (chi khi TRACKING)
            error = 0.0
            if found and cx_filtered >= 0:
                error      = float(cx_filtered - center_x)
                pid_output = pid.update(error, dt if dt > 0 else 0.033)
                neck_abs   = servo_center - pid_output
                neck_abs   = max(servo_min_abs, min(servo_max_abs, neck_abs))
            else:
                pid.reset_memory()

            # 6. Gửi lệnh robot
            if state != RobotState.LOST:
                robot.set_head_angle(pid_output)

            # 7. Terminal log
            fps = 1.0 / dt if dt > 0 else 0.0
            if frame_count % 10 == 0:
                print("[f={:04d}] {} | err={:+.1f}px | PID={:+.2f} | Neck={:.0f}deg | skip={} | fps={:.1f}".format(
                    frame_count,
                    state.name,
                    error,
                    pid_output,
                    neck_abs,
                    skip,
                    fps,
                ))

            # 8. UI (optional)
            if show_ui:
                disp = frame.copy()
                if found and bbox:
                    x, y, wb, hb = bbox
                    cv2.rectangle(disp, (x, y), (x+wb, y+hb), (0, 255, 0), 2)
                    cv2.circle(disp, (int(cx_filtered), cy_raw), 5, (0, 255, 255), -1)
                cv2.line(disp, (center_x, 0), (center_x, H), (255, 0, 0), 1)
                cv2.putText(disp,
                    "{} | skip={} | fps={:.1f}".format(state.name, skip, fps),
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(disp,
                    "err={:+.1f} PID={:+.2f} Neck={:.0f}".format(error, pid_output, neck_abs),
                    (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Yanshee Tracker", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()
        if show_ui:
            cv2.destroyAllWindows()
        print("[INFO] Done. Neck last angle: {:.0f}deg".format(
            robot.get_current_angle()))


if __name__ == "__main__":
    main()