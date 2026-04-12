import cv2
import time
import json
import csv
import os

from core.control import PIDController
from core.filters import TrackerKalmanFilter
from core.state_machine import TrackingStateMachine, RobotState
from hardware.yanshee_interface import YansheeInterface

# Import kiến trúc tối ưu mới cho RPi 3
from core.vision_haarcascade import VisionHaarKCF
from core.adaptive_sheduler import AdaptiveScheduler

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    print("==================================================")
    print(" YANSHEE FACE TRACKING SYSTEM (ADAPTIVE)")
    print("==================================================")

    # Đọc config, fallback an toàn nếu thiếu key
    config = load_config()
    sys_cfg = config.get("system", {"log_csv": True, "csv_file_path": "results/logs/robot_tracking.csv", "use_kalman": True, "use_pid": True})
    cam_cfg = config.get("camera", {"cascade_path": "haarcascade_frontalface_default.xml", "frame_width": 320, "frame_height": 240})
    rob_cfg = config.get("robot_yanshee", {"timeout_lost": 2.0, "max_angle": 180, "min_angle": 0, "default_angle": 90, "backoff_delta": 2})
    pid_cfg = config.get("controller_pid", {"Kp": 0.05, "Ki": 0.0, "Kd": 0.01, "max_integral": 10})
    kf_cfg  = config.get("filter_kalman", {"process_noise_cov": 0.03, "measurement_noise_cov": 0.1})

    # 1. Init Vision & Scheduler
    print("[INFO] Init vision & adaptive scheduler...")
    vision = VisionHaarKCF(cascade_path=cam_cfg.get("cascade_path", "haarcascade_frontalface_default.xml"), skip=5)
    sched = AdaptiveScheduler(enabled=True, base_skip=5, alpha=0.08, beta=0.05)

    pid = PIDController(Kp=pid_cfg["Kp"], Ki=pid_cfg["Ki"], Kd=pid_cfg["Kd"], max_integral=pid_cfg["max_integral"])
    kalman = TrackerKalmanFilter(process_noise=kf_cfg["process_noise_cov"], measurement_noise=kf_cfg["measurement_noise_cov"])
    fsm = TrackingStateMachine(timeout_lost=rob_cfg["timeout_lost"])
    
    # 2. Init Robot (Real hardware)
    print("[INFO] Init robot hardware...")
    robot = YansheeInterface(
        is_simulation=False, 
        max_angle=rob_cfg["max_angle"], 
        min_angle=rob_cfg["min_angle"], 
        default_angle=rob_cfg["default_angle"]
    )

    # 3. Init Camera
    print("[INFO] Init camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return

    # 4. Setup CSV Logging
    csv_file = None
    csv_writer = None
    if sys_cfg["log_csv"]:
        os.makedirs(os.path.dirname(sys_cfg["csv_file_path"]), exist_ok=True)
        csv_file = open(sys_cfg["csv_file_path"], mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "state", "box_x", "box_y", "box_w", "box_h", "filtered_x", "error_angle", "control_angle", "vision_time_ms", "control_time_ms", "fps", "jitter"])

    frame_count = 0
    prev_time = time.time()
    kalman_initialized = False

    print("\n[INFO] System ready. Press CTRL+C to stop.\n")

    try:
        while True:
            loop_start_time = time.time()
            dt = loop_start_time - prev_time
            prev_time = loop_start_time

            ret, frame = cap.read()
            if not ret: continue

            frame_count += 1
            frame_center_x = cam_cfg["frame_width"] // 2

            # ---------------------------------------------------------
            # VISION & ADAPTIVE SCHEDULING
            # ---------------------------------------------------------
            t_vision_start = time.time()
            
            # Quét khuôn mặt kết hợp KCF
            target_found, bbox, center_x, center_y = vision.process(frame)
            
            # Tính toán độ rung nhiễu (Jitter) so với Kalman
            anchor_x = int(kalman.kf.statePost[0, 0]) if kalman_initialized else center_x
            lost = not target_found
            jitter_estimate = abs(center_x - anchor_x) if (target_found and kalman_initialized and center_x >= 0) else 0.0
            
            # Cập nhật số lượng frame sẽ skip cho vòng lặp sau
            new_skip = sched.compute(center_x, center_y, jitter_estimate, lost)
            vision.skip = new_skip

            vision_time_ms = (time.time() - t_vision_start) * 1000

            box_x, box_y, box_w, box_h = (-1, -1, -1, -1)
            if target_found and bbox is not None:
                box_x, box_y, box_w, box_h = bbox
            filtered_x = center_x
            jitter = jitter_estimate

            # ---------------------------------------------------------
            # FILTER & PID CONTROL
            # ---------------------------------------------------------
            t_control_start = time.time()
            if sys_cfg["use_kalman"]:
                if target_found:
                    if not kalman_initialized:
                        kalman.init_state(center_x, center_y)
                        kalman_initialized = True
                    else:
                        filtered_x, _ = kalman.update(center_x, center_y)
                elif kalman_initialized:
                    filtered_x, _ = kalman.predict()
                    
            current_state = fsm.get_state()
            control_output = 0.0
            error_angle = 0.0
            
            if kalman_initialized and current_state == RobotState.TRACKING:
                error_x_pixel = frame_center_x - filtered_x
                error_angle = robot.pixel_to_angle(error_x_pixel, cam_cfg["frame_width"])
                if sys_cfg["use_pid"]:
                    control_output = pid.update(error_angle, dt)

            predicted_angle = robot.get_current_angle() + control_output
            new_state = fsm.update(target_found=target_found, predicted_angle=predicted_angle, max_angle=rob_cfg["max_angle"], min_angle=rob_cfg["min_angle"])
            control_time_ms = (time.time() - t_control_start) * 1000

            # ---------------------------------------------------------
            # SEND COMMAND TO HARDWARE
            # ---------------------------------------------------------
            if new_state == RobotState.SATURATED:
                pid.reset_memory() 
                clamped_angle = max(min(predicted_angle, rob_cfg["max_angle"]), rob_cfg["min_angle"])
                predicted_angle = clamped_angle - rob_cfg["backoff_delta"] if clamped_angle == rob_cfg["max_angle"] else clamped_angle + rob_cfg["backoff_delta"]
                robot.set_head_angle(predicted_angle, new_state.name)
            elif new_state in [RobotState.TRACKING, RobotState.SEARCH]:
                robot.set_head_angle(predicted_angle, new_state.name)
            elif new_state == RobotState.LOST:
                robot.set_head_angle(rob_cfg["default_angle"], new_state.name)
                kalman_initialized = False 

            # ---------------------------------------------------------
            # LOGGING (NO UI)
            # ---------------------------------------------------------
            fps = 1.0 / dt if dt > 0 else 0.0
            if csv_writer:
                csv_writer.writerow([frame_count, new_state.name, box_x, box_y, box_w, box_h, int(filtered_x), round(error_angle, 2), round(control_output, 2), round(vision_time_ms, 2), round(control_time_ms, 2), round(fps, 2), round(jitter, 2)])

            if frame_count % 10 == 0:
                print("[Run] Frame: {:4d} | State: {:10s} | Skip: {:2d} | FPS: {:.1f} | Vision: {:.1f}ms".format(
                    frame_count, new_state.name, new_skip, fps, vision_time_ms))

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")

    finally:
        cap.release()
        if csv_file:
            csv_file.close()
            print("[INFO] Log saved to: {}".format(sys_cfg.get('csv_file_path', 'results/logs/')))
        robot.set_head_angle(rob_cfg["default_angle"], "SHUTDOWN")

if __name__ == "__main__":
    main()