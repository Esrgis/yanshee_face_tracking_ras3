import time

class YansheeInterface:
    """
    Hardware interface & abstraction layer
    """
    def __init__(self, is_simulation=True, max_angle=180.0, min_angle=0.0, default_angle=90.0):
        self.is_simulation = is_simulation
        self.max_angle = float(max_angle)
        self.min_angle = float(min_angle)
        
        # Current angle
        self.current_angle = float(default_angle)
        
        # Camera FOV (Field of View): ~60 degrees
        self.CAMERA_FOV_H = 60.0 

        if self.is_simulation:
            print("[HARDWARE] Running in simulation mode")
        else:
            print("[HARDWARE] Connecting to Yanshee...")
            # TODO: Init SDK or serial port

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
        
        # Map pixel to angle
        error_angle = (error_x * self.CAMERA_FOV_H) / frame_width
        return error_angle

    def set_head_angle(self, target_angle, current_state_name="TRACKING"):
        # Send angle command to motor
        # Clamp angle to safe range
        safe_angle = max(min(target_angle, self.max_angle), self.min_angle)
        
        # Update current angle
        self.current_angle = safe_angle

        # Send command
        if self.is_simulation:
            # Simulation mode
            print(f"[SERVO] State: {current_state_name:10} | Angle: {safe_angle:5.1f}°")
        else:
            # TODO: Send command to motor
            pass

    def get_current_angle(self):
        # Get current motor angle
        return self.current_angle