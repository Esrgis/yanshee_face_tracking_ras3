class PIDController:
    """
    Bộ điều khiển PID chuẩn công nghiệp có tích hợp cơ chế Anti-Windup.
    Được thiết kế để chạy trong vòng lặp thời gian thực với dt biến thiên.
    """
    def __init__(self, Kp=0.05, Ki=0.0, Kd=0.01, max_integral=50.0):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        
        # Giới hạn an toàn để chống hiện tượng Bão hòa tích phân (Windup)
        self.max_integral = float(max_integral)
        
        # Bộ nhớ của PID
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """
        Tính toán tín hiệu điều khiển dựa trên sai số và thời gian lấy mẫu.
        
        :param error: Sai số hiện tại (Ví dụ: Góc đích - Góc hiện tại)
        :param dt: Thời gian trôi qua kể từ lần gọi trước (Delta time) tính bằng giây.
        :return: Lượng thay đổi góc/vận tốc cần thiết (Control Output)
        """
        # Bảo vệ chia cho 0 hoặc dt âm (do lỗi hệ thống)
        if dt <= 0.0:
            return 0.0

        # 1. Khâu TỈ LỆ (Proportional)
        P_out = self.Kp * error

        # 2. Khâu TÍCH PHÂN (Integral) - Kèm Anti-Windup [cite: 85, 91]
        self.integral += error * dt
        # Kẹp cứng giá trị tích phân không cho phình to quá max_integral
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        I_out = self.Ki * self.integral

        # 3. Khâu ĐẠO HÀM (Derivative) [cite: 7]
        derivative = (error - self.prev_error) / dt
        D_out = self.Kd * derivative

        # Lưu lại sai số cho chu kỳ tiếp theo
        self.prev_error = error

        # Tổng hợp tín hiệu điều khiển
        output = P_out + I_out + D_out
        return output

    def reset_memory(self):
        """
        HÀM CẤP CỨU (Dùng cho State Machine) [cite: 92, 114, 138]
        Xóa sạch bộ nhớ của PID (đặc biệt là khâu Tích phân).
        Phải được gọi ngay lập tức khi hệ thống chuyển sang trạng thái SATURATED.
        """
        self.integral = 0.0
        self.prev_error = 0.0
        # print("[PID SYSTEM] Đã xả toàn bộ lỗi Tích phân (Anti-windup triggered)!")

    def set_tunings(self, Kp, Ki, Kd):
        """
        Cập nhật hệ số nóng (Dùng khi chạy kịch bản Auto-Tuning).
        """
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)