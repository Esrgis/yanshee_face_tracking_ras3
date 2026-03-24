import numpy as np
import cv2

class TrackerKalmanFilter:
    """
    Bộ lọc Kalman đóng vai trò 'nhà tiên tri'.
    Lọc nhiễu (Jitter) từ Vision và dự đoán vị trí khi mất dấu.
    """
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        # 4 trạng thái (x, y, vx, vy), 2 đo lường (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Ma trận đo lường (Hệ thống chỉ đo được vị trí x, y từ YOLO)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Ma trận chuyển trạng thái (Mô hình động lực học cơ bản: vị trí mới = vị trí cũ + vận tốc * dt)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Ma trận hiệp phương sai nhiễu
        # processNoiseCov: Độ tin cậy vào mô hình vật lý (Càng nhỏ càng tin mô hình)
        # measurementNoiseCov: Độ tin cậy vào Camera (Càng nhỏ càng tin Camera)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)

    def init_state(self, x, y):
        """Khởi tạo trạng thái ban đầu ngay khoảnh khắc YOLO bắt được mặt"""
        self.kf.statePre = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
        self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)

    def predict(self):
        """
        Dự đoán vị trí tiếp theo. 
        Sức mạnh thực sự: Gọi hàm này ngay cả khi YOLO bị mù/mất frame để lấy tọa độ bù vào!
        """
        prediction = self.kf.predict()
        return int(prediction[0, 0]), int(prediction[1, 0])

    def update(self, x, y):
        """
        Cập nhật tọa độ thực tế từ Camera vào bộ lọc để điều chỉnh lại dự đoán.
        """
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        
        # Trả về tọa độ đã được lọc sạch nhiễu
        filtered_state = self.kf.statePost
        return int(filtered_state[0, 0]), int(filtered_state[1, 0])

class MovingAverage:
    """
    Bộ lọc trung bình động (EMA - Exponential Moving Average).
    Dùng để làm mượt thêm tín hiệu sai số trước khi nhét vào PID nếu cần.
    """
    def __init__(self, alpha=0.8):
        self.alpha = float(alpha)
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_val
        return self.value