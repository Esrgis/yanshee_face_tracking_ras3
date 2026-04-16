import time
import threading

try:
    import queue
except ImportError:
    import Queue as queue # Dành cho Python 2/3.5 compatibility

class YansheeInterface:
    def __init__(self, config_dict, is_simulation=True):
        self.is_simulation = is_simulation
        self.use_thread = config_dict.get("use_multithreading", True)
        self.current_angle = 0.0
        
        # Thiết lập hàng đợi (Queue)
        if self.use_thread:
            # maxsize=1 là chìa khóa chống delay tích lũy
            self.angle_queue = queue.Queue(maxsize=1) 
            self.thread = threading.Thread(target=self._servo_worker)
            self.thread.daemon = True # Thread tự chết khi chương trình chính tắt
            self.thread.start()
            print("[HARDWARE] Started in MULTI-THREAD mode")
        else:
            print("[HARDWARE] Started in SINGLE-THREAD mode")

    def set_head_angle(self, target_angle):
        safe_angle = max(min(target_angle, 45), -45)
        
        if self.use_thread:
            # --- ĐA LUỒNG ---
            # Nếu hộp thư đã có lệnh cũ chưa kịp gửi, ném nó đi
            if self.angle_queue.full():
                try: self.angle_queue.get_nowait()
                except queue.Empty: pass
            # Bỏ lệnh mới nhất vào hộp thư
            self.angle_queue.put(safe_angle)
        else:
            # --- ĐƠN LUỒNG ---
            self._send_to_hardware(safe_angle)
            
    def _servo_worker(self):
        """Luồng phụ chuyên chầu chực lấy góc từ Queue để gửi I/O"""
        while True:
            # Chờ đến khi có lệnh trong hàng đợi
            angle = self.angle_queue.get() 
            self._send_to_hardware(angle)
            self.angle_queue.task_done()

    def _send_to_hardware(self, angle):
        """Hàm giao tiếp Yanshee API (Hàm chặn / Blocking Call)"""
        self.current_angle = angle
        if self.is_simulation:
            # print giả lập
            pass
        else:
            # TODO: Gọi API gửi xuống motor (Tốn 100ms ở đây)
            pass