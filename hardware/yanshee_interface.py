import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue  # type: ignore

try:
    import YanAPI
except ImportError as e:
    print("[WARNING] Không tìm thấy YanAPI. Bắt buộc chạy ở chế độ simulation.")

class YansheeInterface:
    def __init__(self, config_dict, is_simulation=True):
        self.is_simulation = is_simulation
        self.use_thread = config_dict.get("use_multithreading", True)
        
        # --- CẬP NHẬT THÔNG SỐ TỪ YANAPI DOCS ---
        self.servo_name = "NeckLR" 
        self.center_angle = 90.0  # Góc nhìn thẳng
        self.min_abs_angle = 15.0 # Giới hạn quay trái
        self.max_abs_angle = 165.0 # Giới hạn quay phải
        
        self.current_angle = self.center_angle
        
        if self.use_thread:
            self.angle_queue = queue.Queue(maxsize=1) 
            self.thread = threading.Thread(target=self._servo_worker)
            self.thread.daemon = True
            self.thread.start()
            print("[HARDWARE] Started in MULTI-THREAD mode")
        else:
            print("[HARDWARE] Started in SINGLE-THREAD mode")

    def set_head_angle(self, target_pid_angle):
        """
        Nhận góc từ bộ PID (ví dụ: -30 đến +30 độ)
        Biến đổi thành góc vật lý tuyệt đối (ví dụ: 60 đến 120 độ)
        """
        # 1. Chuyển đổi góc tương đối thành góc tuyệt đối trên servo
        absolute_angle = self.center_angle - target_pid_angle # Trừ hay cộng tùy chiều camera của bạn
        
        # 2. Kẹp (Clamp) góc vào giới hạn an toàn của YanAPI (15 - 165)
        safe_angle = max(min(absolute_angle, self.max_abs_angle), self.min_abs_angle)
        
        # 3. Gửi xuống luồng
        if self.use_thread:
            if self.angle_queue.full():
                try: self.angle_queue.get_nowait()
                except queue.Empty: pass
            self.angle_queue.put(safe_angle)
        else:
            self._send_to_hardware(safe_angle)
            
    def _servo_worker(self):
        while True:
            angle = self.angle_queue.get() 
            self._send_to_hardware(angle)
            self.angle_queue.task_done()

    def _send_to_hardware(self, angle):
        self.current_angle = angle
        
        # YanAPI yêu cầu góc phải là số nguyên (int)
        final_angle_int = int(angle)
        
        if self.is_simulation:
            # print(f"[SIMULATION] Xoay cổ (NeckLR) tới: {final_angle_int} độ")
            pass
        else:
            try:
                # Gửi cấu trúc Dict chuẩn của YanAPI với runtime thấp nhất là 200ms
                YanAPI.set_servos_angles({self.servo_name: final_angle_int}, 200)
            except Exception as e:
                print("[ERROR] YanAPI failed: {}".format(e))