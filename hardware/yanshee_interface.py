import time

class YansheeInterface:
    """
    Lớp giao tiếp phần cứng (Hardware Abstraction Layer - HAL).
    Cách ly hoàn toàn thuật toán điều khiển với thiết bị vật lý.
    Cho phép test offline (Mocking) trên Laptop không cần robot thật.
    """
    def __init__(self, is_simulation=True, max_angle=180.0, min_angle=0.0, default_angle=90.0):
        self.is_simulation = is_simulation
        self.max_angle = float(max_angle)
        self.min_angle = float(min_angle)
        
        # Trạng thái vật lý hiện tại của cổ robot
        self.current_angle = float(default_angle)
        
        # Góc nhìn ngang của Camera (Field of View). 
        # Cần đo lại thông số thực tế của camera Yanshee, giả định tạm là 60 độ.
        self.CAMERA_FOV_H = 60.0 

        if self.is_simulation:
            print("[HARDWARE INIT] Đang chạy chế độ GIẢ LẬP (Mocking) trên Laptop.")
        else:
            print("[HARDWARE INIT] Đang kết nối với Cổng Serial/SDK của Yanshee...")
            # TODO: Viết code khởi tạo SDK/Serial port ở đây
            # Ví dụ: self.robot = YansheeAPI()

    def pixel_to_angle(self, error_x, frame_width):
        """
        BƯỚC TIỀN XỬ LÝ SỐNG CÒN:
        Chuyển đổi sai số từ Pixel sang Góc (Độ).
        Giúp hệ số PID (Kp, Ki, Kd) giữ nguyên ý nghĩa vật lý dù đổi độ phân giải camera.
        
        :param error_x: Sai số vị trí x tính bằng pixel.
        :param frame_width: Chiều rộng khung hình camera (Ví dụ: 320).
        :return: Sai số tính bằng độ.
        """
        if frame_width <= 0:
            return 0.0
        
        # Công thức ánh xạ tuyến tính cơ bản từ FOV
        error_angle = (error_x * self.CAMERA_FOV_H) / frame_width
        return error_angle

    def set_head_angle(self, target_angle, current_state_name="TRACKING"):
        """
        Lệnh duy nhất trong toàn bộ hệ thống được phép tác động đến động cơ.
        
        :param target_angle: Góc đích muốn quay tới.
        :param current_state_name: Tên của State Machine (Để in log debug).
        """
        # 1. Bảo vệ phần cứng lần cuối (Hard-Clamp)
        # Bất chấp PID hay Kalman tính ra cái gì, không bao giờ cho phép vượt max/min
        safe_angle = max(min(target_angle, self.max_angle), self.min_angle)
        
        # Cập nhật góc hiện tại của robot
        self.current_angle = safe_angle

        # 2. Xử lý xuất lệnh
        if self.is_simulation:
            # Chế độ Mock: Tuôn log ra Terminal để quan sát luồng FSM
            print(f"[SERVO MOCK] State: {current_state_name:10} | Lệnh quay thực tế: {safe_angle:5.1f}°")
        else:
            # TODO: Đẩy lệnh `safe_angle` xuống SDK/Motor thật của Yanshee
            # Ví dụ: self.robot.set_servo_angle(id=1, angle=safe_angle)
            pass

    def get_current_angle(self):
        """
        Trả về góc hiện tại để thuật toán PID có gốc tính toán.
        """
        return self.current_angle