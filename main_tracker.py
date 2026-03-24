import cv2
import time
import json
import csv
import math
import os

# Import các module lõi anh vừa viết
from core.vision import VisionSystem
from core.control import PIDController
from core.filters import TrackerKalmanFilter, MovingAverage
from core.state_machine import TrackingStateMachine, RobotState
from hardware.yanshee_interface import YansheeInterface

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    print("==================================================")
    print(" KHỞI ĐỘNG HỆ THỐNG VISUAL TRACKING ROBOT YANSHEE ")
    print("==================================================")

    # 1. ĐỌC CẤU HÌNH
    config = load_config()
    sys_cfg = config["system"]
    cam_cfg = config["camera"]
    rob_cfg = config["robot_yanshee"]
    pid_cfg = config["controller_pid"]
    kf_cfg = config["filter_kalman"]
    env_cfg = config["testing_env"]

    # 2. KHỞI TẠO CÁC MODULE (5 Viên ngọc vô cực)
    vision = VisionSystem(model_path=cam_cfg["model_path"], conf_threshold=cam_cfg["conf_threshold"])
    
    pid = PIDController(Kp=pid_cfg["Kp"], Ki=pid_cfg["Ki"], Kd=pid_cfg["Kd"], max_integral=pid_cfg["max_integral"])
    
    kalman = TrackerKalmanFilter(process_noise=kf_cfg["process_noise_cov"], measurement_noise=kf_cfg["measurement_noise_cov"])
    
    fsm = TrackingStateMachine(timeout_lost=rob_cfg["timeout_lost"])
    
    robot = YansheeInterface(
        is_simulation=not env_cfg["use_live_camera"], 
        max_angle=rob_cfg["max_angle"], 
        min_angle=rob_cfg["min_angle"], 
        default_angle=rob_cfg["default_angle"]
    )

    # 3. KHỞI TẠO LUỒNG VIDEO
    if env_cfg["use_live_camera"]:
        cap = cv2.VideoCapture(0) # Webcam
    else:
        cap = cv2.VideoCapture(env_cfg["video_source"])
    
    if not cap.isOpened():
        print(f"[FATAL ERROR] Không thể mở luồng video từ: {env_cfg['video_source']}")
        return

    # 4. CHUẨN BỊ GHI LOG CSV
    csv_file = None
    csv_writer = None
    if sys_cfg["log_csv"]:
        os.makedirs(os.path.dirname(sys_cfg["csv_file_path"]), exist_ok=True)
        csv_file = open(sys_cfg["csv_file_path"], mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Ghi Header
        csv_writer.writerow(["frame", "state", "target_x", "error_angle", "control_angle", "latency_ms", "fps", "jitter"])

    # --- CÁC BIẾN TOÀN CỤC CHO VÒNG LẶP ---
    frame_count = 0
    prev_time = time.time()
    prev_center_x = -1
    kalman_initialized = False

    print("\n[SYSTEM] HỆ THỐNG ĐÃ SẴN SÀNG. BẮT ĐẦU VÒNG LẶP THỜI GIAN THỰC!")
    print("Nhấn 'q' trên cửa sổ hình ảnh để thoát.\n")

    # ==========================================================
    # VÒNG LẶP CHÍNH (THE HEARTBEAT)
    # ==========================================================
    while True:
        loop_start_time = time.time()
        
        # Tính delta time (dt) thực tế cho PID
        dt = loop_start_time - prev_time
        prev_time = loop_start_time

        ret, frame = cap.read()
        if not ret:
            print("[INFO] Đã hết video hoặc mất kết nối camera.")
            break
        
        frame = cv2.resize(frame, (cam_cfg["frame_width"], cam_cfg["frame_height"]))
        frame_count += 1
        frame_center_x = cam_cfg["frame_width"] // 2

        # --- BƯỚC A: THỊ GIÁC (VISION) ---
        target_found, bbox, center_x, center_y = vision.process_frame(frame)
        
        filtered_x = center_x
        jitter = 0.0

        # --- BƯỚC B: LỌC NHIỄU & DỰ ĐOÁN (KALMAN) ---
        if sys_cfg["use_kalman"]:
            if target_found:
                if not kalman_initialized:
                    kalman.init_state(center_x, center_y)
                    kalman_initialized = True
                else:
                    filtered_x, _ = kalman.update(center_x, center_y)
            elif kalman_initialized:
                # Bị che khuất? Kalman sẽ nhắm mắt đoán bừa vị trí tiếp theo!
                filtered_x, _ = kalman.predict()
                
            # Tính Jitter (Sự rung lắc giữa tọa độ YOLO đo được và tọa độ Kalman đã lọc)
            if target_found and center_x != -1:
                jitter = abs(center_x - filtered_x)

        # --- BƯỚC C: TÍNH SAI SỐ VÀ ĐIỀU KHIỂN (PID) ---
        current_state = fsm.get_state()
        control_output = 0.0
        error_angle = 0.0
        
        if kalman_initialized and current_state == RobotState.TRACKING:
            # 1. Tính sai số Pixel
            error_x_pixel = frame_center_x - filtered_x
            
            # 2. Ánh xạ Pixel -> Góc
            error_angle = robot.pixel_to_angle(error_x_pixel, cam_cfg["frame_width"])
            
            # 3. Chạy PID
            if sys_cfg["use_pid"]:
                control_output = pid.update(error_angle, dt)

        # Dự đoán góc tiếp theo của phần cứng
        predicted_angle = robot.get_current_angle() + control_output

        # --- BƯỚC D: CẬP NHẬT MÁY TRẠNG THÁI (STATE MACHINE) ---
        new_state = fsm.update(
            target_found=target_found, 
            predicted_angle=predicted_angle, 
            max_angle=rob_cfg["max_angle"], 
            min_angle=rob_cfg["min_angle"]
        )

        # --- BƯỚC E: XỬ LÝ SỰ KIỆN TRẠNG THÁI & XUẤT LỆNH ---
        if new_state == RobotState.SATURATED:
            pid.reset_memory() # Cấp cứu Anti-windup!
            
            # Kẹp cứng góc và lui nhẹ (Back-off)
            clamped_angle = max(min(predicted_angle, rob_cfg["max_angle"]), rob_cfg["min_angle"])
            if clamped_angle == rob_cfg["max_angle"]:
                predicted_angle = clamped_angle - rob_cfg["backoff_delta"]
            else:
                predicted_angle = clamped_angle + rob_cfg["backoff_delta"]
                
            robot.set_head_angle(predicted_angle, new_state.name)

        elif new_state == RobotState.TRACKING:
            robot.set_head_angle(predicted_angle, new_state.name)
            
        elif new_state == RobotState.SEARCH:
            # Ở đây có thể nhúng logic xoay camera quét xung quanh
            pass
            
        elif new_state == RobotState.LOST:
            # Reset về giữa
            robot.set_head_angle(rob_cfg["default_angle"], new_state.name)
            kalman_initialized = False # Xóa trí nhớ Kalman

        # --- BƯỚC F: ĐO LƯỜNG HIỆU NĂNG & GHI LOG ---
        latency_ms = (time.time() - loop_start_time) * 1000
        fps = 1.0 / dt if dt > 0 else 0.0

        if csv_writer:
            csv_writer.writerow([frame_count, new_state.name, filtered_x, round(error_angle, 2), round(control_output, 2), round(latency_ms, 2), round(fps, 2), round(jitter, 2)])

        # --- BƯỚC G: HIỂN THỊ LÊN MÀN HÌNH ---
        # Vẽ Bounding Box YOLO/KCF (Màu Xanh lá)
        if bbox is not None and target_found:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

        # Vẽ tâm Kalman dự đoán (Màu Đỏ)
        if kalman_initialized:
            cv2.circle(frame, (int(filtered_x), cam_cfg["frame_height"]//2), 6, (0, 0, 255), -1)

        # In thông số lên màn hình
        cv2.putText(frame, f"State: {new_state.name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Angle: {robot.get_current_angle():.1f} deg", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Yanshee Visual Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()
        print(f"\n[INFO] Đã lưu dữ liệu log tại: {sys_cfg['csv_file_path']}")

if __name__ == "__main__":
    main()