import cv2
import time
import json
import csv
import math
import os

from core.vision_onnx import VisionONNX
from core.vision_ultralytics import VisionUltralytics
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

    # 2. KHỞI TẠO CÁC MODULE 
    backend = cam_cfg.get("model_backend", "onnx")
    if backend == "onnx":
        vision = VisionONNX(model_path=cam_cfg["model_path"], conf_threshold=cam_cfg["conf_threshold"])
    elif backend == "pt":
        vision = VisionUltralytics(model_path=cam_cfg["model_path"], conf_threshold=cam_cfg["conf_threshold"])
    else:
        raise ValueError(f"[CONFIG ERROR] model_backend không hợp lệ: {backend}")
    
    pid = PIDController(Kp=pid_cfg["Kp"], Ki=pid_cfg["Ki"], Kd=pid_cfg["Kd"], max_integral=pid_cfg["max_integral"])
    kalman = TrackerKalmanFilter(process_noise=kf_cfg["process_noise_cov"], measurement_noise=kf_cfg["measurement_noise_cov"])
    fsm = TrackingStateMachine(timeout_lost=rob_cfg["timeout_lost"])
    
    # 3. KHỞI TẠO LUỒNG VIDEO VÀ HARDWARE MOCK
    source_type = env_cfg.get("source_type", "video").lower()
    
    robot = YansheeInterface(
        is_simulation=(source_type == "video"), # Nếu chạy file video thì chắc chắn là giả lập
        max_angle=rob_cfg["max_angle"], 
        min_angle=rob_cfg["min_angle"], 
        default_angle=rob_cfg["default_angle"]
    )
    
    if source_type == "ip_camera":  # <--- ĐÂY LÀ PHẦN MỚI THÊM VÀO!
        ip_url = env_cfg.get("ip_camera_url", "")
        cap = cv2.VideoCapture(ip_url)
        print(f"[INFO] Kích hoạt chế độ IP CAMERA (Robot Wi-Fi): {ip_url}")

    elif source_type == "webcam":
        cam_index = env_cfg.get("webcam_index", 0)
        cap = cv2.VideoCapture(cam_index)
        print(f"[INFO] Kích hoạt chế độ WEBCAM (Device Index: {cam_index})")
        
    elif source_type == "hdmi":
        hdmi_index = env_cfg.get("hdmi_index", 1)
        cap = cv2.VideoCapture(hdmi_index, cv2.CAP_DSHOW)
        print(f"[INFO] Kích hoạt chế độ tham chiếu HDMI (Device Index: {hdmi_index})")
        
    elif source_type == "video":
        video_path = env_cfg.get("video_path", "")
        cap = cv2.VideoCapture(video_path)
        print(f"[INFO] Kích hoạt chế độ VIDEO FILE: {video_path}")
        
        
    else:
        raise ValueError(f"[FATAL ERROR] Cấu hình source_type không hợp lệ: '{source_type}'. Chỉ chấp nhận: 'video', 'webcam', 'hdmi', 'ip_camera'.")
    # Kiểm tra xem có mở được luồng ảnh không
    if not cap.isOpened():
        print(f"[FATAL ERROR] Không thể mở luồng video từ nguồn: {source_type.upper()}")
        print(" -> Mẹo: Nếu dùng HDMI/Webcam, thử đổi index (0, 1, 2...) trong file config.json.")
        return

    # 4. CHUẨN BỊ GHI LOG CSV (Đã sửa lại Header 13 cột chuẩn)
    csv_file = None
    csv_writer = None
    if sys_cfg["log_csv"]:
        os.makedirs(os.path.dirname(sys_cfg["csv_file_path"]), exist_ok=True)
        csv_file = open(sys_cfg["csv_file_path"], mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "state", "box_x", "box_y", "box_w", "box_h", "filtered_x", "error_angle", "control_angle", "vision_time_ms", "control_time_ms", "fps", "jitter"])

    # --- CÁC BIẾN TOÀN CỤC CHO VÒNG LẶP ---
    frame_count = 0
    prev_time = time.time()
    kalman_initialized = False
    search_frame_counter = 0
    
    print("\n[SYSTEM] HỆ THỐNG ĐÃ SẴN SÀNG. BẮT ĐẦU VÒNG LẶP THỜI GIAN THỰC!")
    print("Nhấn 'q' trên cửa sổ hình ảnh để thoát.\n")

    # ==========================================================
    # VÒNG LẶP CHÍNH
    # ==========================================================
    while True:
        loop_start_time = time.time()
        
        # Tính delta time tổng
        dt = loop_start_time - prev_time
        prev_time = loop_start_time

        ret, frame = cap.read()
        if not ret:
            print("[INFO] Đã hết video hoặc mất kết nối camera.")
            break
        
        frame = cv2.resize(frame, (cam_cfg["frame_width"], cam_cfg["frame_height"]))
        frame_count += 1
        frame_center_x = cam_cfg["frame_width"] // 2

       # --- BƯỚC A: THỊ GIÁC (VISION INFERENCE) ---
        t_vision_start = time.time()
        
        anchor_x, anchor_y = -1, -1
        if kalman_initialized:
            pred_x, pred_y = kalman.predict() 
            anchor_x, anchor_y = int(pred_x), int(pred_y)

        # CHỈ GỌI YOLO (QUÉT NẶNG) NẾU ĐANG TRACKING, HOẶC CỨ MỖI 3 FRAME LÚC ĐANG SEARCH
        if fsm.get_state() == RobotState.SEARCH:
            search_frame_counter += 1
            if search_frame_counter % 3 == 0:
                target_found, bbox, center_x, center_y = vision.process_frame(frame, prev_x=anchor_x, prev_y=anchor_y)
            else:
                # Bỏ qua không gọi Vision để giảm tải CPU, giữ nguyên trạng thái không tìm thấy
                target_found, bbox, center_x, center_y = False, None, -1, -1
        else:
            # Lúc bình thường (TRACKING hoặc LOST) thì cứ chạy bình thường
            target_found, bbox, center_x, center_y = vision.process_frame(frame, prev_x=anchor_x, prev_y=anchor_y)
            search_frame_counter = 0 # Reset bộ đếm

        vision_time_ms = (time.time() - t_vision_start) * 1000
        
        box_x, box_y, box_w, box_h = -1, -1, -1, -1
        if target_found and bbox is not None:
            box_x, box_y, box_w, box_h = bbox

        filtered_x = center_x
        jitter = 0.0

        # --- BƯỚC B: LỌC & ĐIỀU KHIỂN (KALMAN + PID + FSM) ---
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

        new_state = fsm.update(
            target_found=target_found, 
            predicted_angle=predicted_angle, 
            max_angle=rob_cfg["max_angle"], 
            min_angle=rob_cfg["min_angle"]
        )
        
        control_time_ms = (time.time() - t_control_start) * 1000

        # --- BƯỚC C: XUẤT LỆNH HARDWARE ---
        if new_state == RobotState.SATURATED:
            pid.reset_memory() 
            clamped_angle = max(min(predicted_angle, rob_cfg["max_angle"]), rob_cfg["min_angle"])
            if clamped_angle == rob_cfg["max_angle"]:
                predicted_angle = clamped_angle - rob_cfg["backoff_delta"]
            else:
                predicted_angle = clamped_angle + rob_cfg["backoff_delta"]
            robot.set_head_angle(predicted_angle, new_state.name)

        elif new_state == RobotState.TRACKING:
            robot.set_head_angle(predicted_angle, new_state.name)
            
        elif new_state == RobotState.SEARCH:
            # Đã sửa lỗi 'pass'. Giữ nguyên góc hiện tại để quét
            robot.set_head_angle(predicted_angle, new_state.name)
            
        elif new_state == RobotState.LOST:
            robot.set_head_angle(rob_cfg["default_angle"], new_state.name)
            kalman_initialized = False 

        # --- BƯỚC D: ĐO LƯỜNG TỔNG & GHI LOG ---
        fps = 1.0 / dt if dt > 0 else 0.0

        if csv_writer:
            csv_writer.writerow([
                frame_count, new_state.name, 
                box_x, box_y, box_w, box_h, 
                int(filtered_x), round(error_angle, 2), round(control_output, 2), 
                round(vision_time_ms, 2), round(control_time_ms, 2), 
                round(fps, 2), round(jitter, 2)
            ])

        # --- BƯỚC E: HIỂN THỊ UI ---
        if bbox is not None and target_found:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

        if kalman_initialized:
            cv2.circle(frame, (int(filtered_x), cam_cfg["frame_height"]//2), 6, (0, 0, 255), -1)

        cv2.putText(frame, f"State: {new_state.name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Angle: {robot.get_current_angle():.1f} deg", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # Sửa lại text hiển thị để anh thấy rõ thời gian của từng khâu
        cv2.putText(frame, f"FPS: {fps:.1f} | Vision: {vision_time_ms:.1f}ms | Ctrl: {control_time_ms:.1f}ms", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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