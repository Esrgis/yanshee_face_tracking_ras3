import time
from enum import Enum

class RobotState(Enum):
    TRACKING = 1    # Đang bám sát bình thường
    SATURATED = 2   # Chạm giới hạn biên cơ khí 
    SEARCH = 3      # Tìm kiếm cục bộ quanh vị trí mất dấu
    LOST = 4        # Mất dấu hoàn toàn 

class TrackingStateMachine:
    def __init__(self, timeout_lost=3.0):
        """
        Khởi tạo Máy trạng thái.
        :param timeout_lost: Thời gian (giây) tối đa cho phép ở trạng thái SEARCH trước khi bỏ cuộc.
        """
        self.current_state = RobotState.SEARCH
        self.lost_start_time = time.time()
        self.timeout_lost = timeout_lost

    def update(self, target_found, predicted_angle, max_angle, min_angle):
        """
        Hàm này được gọi liên tục mỗi frame để đánh giá và chuyển đổi trạng thái.
        
        :param target_found: Bolean - YOLO/KCF có đang khóa được mặt không?
        :param predicted_angle: Float - Góc quay dự kiến nếu tiếp tục chạy PID.
        :param max_angle: Float - Giới hạn vật lý trên.
        :param min_angle: Float - Giới hạn vật lý dưới.
        :return: RobotState - Trạng thái mới của hệ thống.
        """
        
        # ---------------------------------------------------------
        # 1. ĐANG BÁM SÁT (TRACKING)
        # ---------------------------------------------------------
        if self.current_state == RobotState.TRACKING:
            if not target_found:
                self.current_state = RobotState.SEARCH
                self.lost_start_time = time.time() # Bắt đầu bấm giờ
                
            elif predicted_angle >= max_angle or predicted_angle <= min_angle:
                # Xung kích (Edge-trigger) chuyển sang trạng thái bão hòa
                self.current_state = RobotState.SATURATED

        # ---------------------------------------------------------
        # 2. TRẠNG THÁI BÃO HÒA (SATURATED)
        # ---------------------------------------------------------
        elif self.current_state == RobotState.SATURATED:
            # LƯU Ý KỸ THUẬT:
            # Trạng thái này chỉ tồn tại đúng 1 vòng lặp (1 tick) để main_tracker 
            # kịp thời gọi hàm xóa PID (Anti-windup) và lùi lại (Back-off).
            # Ngay sau đó, nó tự động bị ép chuyển sang trạng thái SEARCH.
            self.current_state = RobotState.SEARCH
            self.lost_start_time = time.time()

        # ---------------------------------------------------------
        # 3. ĐANG TÌM KIẾM CỤC BỘ (SEARCH)
        # ---------------------------------------------------------
        elif self.current_state == RobotState.SEARCH:
            if target_found:
                self.current_state = RobotState.TRACKING
            elif (time.time() - self.lost_start_time) > self.timeout_lost:
                self.current_state = RobotState.LOST

        # ---------------------------------------------------------
        # 4. ĐÃ MẤT DẤU (LOST)
        # ---------------------------------------------------------
        elif self.current_state == RobotState.LOST:
            if target_found:
                # Phục hồi thành công khi mục tiêu vô tình xuất hiện lại
                self.current_state = RobotState.TRACKING

        return self.current_state

    def get_state(self):
        return self.current_state
    
    def get_state_name(self):
        return self.current_state.name