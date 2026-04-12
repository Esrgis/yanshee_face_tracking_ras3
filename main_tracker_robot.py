import cv2
import time
import json
import csv
import os

from core.vision_onnx import VisionONNX
from core.control import PIDController
from core.filters import TrackerKalmanFilter
from core.state_machine import TrackingStateMachine, RobotState
from hardware.yanshee_interface import YansheeInterface

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    print("==================================================")
    print(" YANSHEE FACE TRACKING SYSTEM")
    print("==================================================")

    config = load_config()
    sys_cfg = config["system"]
    cam_cfg = config["camera"]
    rob_cfg = config["robot_yanshee"]
    pid_cfg = config["controller_pid"]
    kf_cfg = config["filter_kalman"]

    # 1. Init modules
    print("[INFO] Init vision...")
    vision = VisionONNX(model_path=cam_cfg["model_path"], conf_threshold=cam_cfg["conf_threshold"])
    
    pid = PIDController(Kp=pid_cfg["Kp"], Ki=pid_cfg["Ki"], Kd=pid_cfg["Kd"], max_integral=pid_cfg["max_integral"])
    kalman = TrackerKalmanFilter(process_noise=kf_cfg["process_noise_cov"], measurement_noise=kf_cfg["measurement_noise_cov"])
    fsm = TrackingStateMachine(timeout_lost=rob_cfg["timeout_lost"])
    
    # Use real hardware (not simulation)
    print("[INFO] Init robot...")
    robot = YansheeInterface(
        is_simulation=False, 
        max_angle=rob_cfg["max_angle"], 
        min_angle=rob_cfg["min_angle"], 
        default_angle=rob_cfg["default_angle"]
    )

    # 2. Init camera
    print("[INFO] Init camera...")
    cap = cv2.VideoCapture(0)
    # Set resolution at hardware
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return

    # 3. Setup CSV logging
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
    search_frame_counter = 0

    print("\n[INFO] System ready. Press CTRL+C to stop.\n")

    try:
        while True:
            loop_start_time = time.time()
            dt = loop_start_time - prev_time
            prev_time = loop_start_time

            ret, frame = cap.read()
            if not ret: continue
            
            # Flip frame if needed: frame = cv2.flip(frame, 1) 

            frame_count += 1
            frame_center_x = cam_cfg["frame_width"] // 2

            # Vision
            t_vision_start = time.time()
            anchor_x, anchor_y = -1, -1
            if kalman_initialized:
                pred_x, pred_y = kalman.predict() 
                anchor_x, anchor_y = int(pred_x), int(pred_y)

            # Skip frames if searching
            if fsm.get_state() == RobotState.SEARCH:
                search_frame_counter += 1
                if search_frame_counter % 3 == 0:
                    target_found, bbox, center_x, center_y = vision.process_frame(frame, prev_x=anchor_x, prev_y=anchor_y)
                else:
                    target_found, bbox, center_x, center_y = False, None, -1, -1
            else:
                target_found, bbox, center_x, center_y = vision.process_frame(frame, prev_x=anchor_x, prev_y=anchor_y)
                search_frame_counter = 0 

            vision_time_ms = (time.time() - t_vision_start) * 1000
            
            box_x, box_y, box_w, box_h = -1, -1, -1, -1
            if target_found and bbox is not None:
                box_x, box_y, box_w, box_h = bbox
            filtered_x = center_x
            jitter = 0.0

            # Filter & Control
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
                    
                if target_found and center_x != -1:
                    jitter = abs(center_x - filtered_x)

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

            # Send command to hardware
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

            # Logging (no UI)
            fps = 1.0 / dt if dt > 0 else 0.0
            if csv_writer:
                csv_writer.writerow([frame_count, new_state.name, box_x, box_y, box_w, box_h, int(filtered_x), round(error_angle, 2), round(control_output, 2), round(vision_time_ms, 2), round(control_time_ms, 2), round(fps, 2), round(jitter, 2)])

            # Print log every 10 frames
            if frame_count % 10 == 0:
                print("[Run] Frame: {0} | State: {1} | FPS: {2:.1f} | Vision: {3:.1f}ms".format(frame_count, new_state.name, fps, vision_time_ms))

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")

    finally:
        # Cleanup
        cap.release()
        if csv_file:
            csv_file.close()
            print(f"[INFO] Log saved to: {sys_cfg['csv_file_path']}")
        # Return to default position
        robot.set_head_angle(rob_cfg["default_angle"], "SHUTDOWN")

if __name__ == "__main__":
    main()