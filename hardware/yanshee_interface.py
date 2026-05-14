# -*- coding: utf-8 -*-
import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue  # type: ignore

try:
    import YanAPI
except ImportError:
    print("[WARNING] YanAPI not found. Forced simulation mode.")

class YansheeInterface:
    def __init__(self, config_dict, is_simulation=True):
        self.is_simulation = is_simulation
        self.use_thread = config_dict.get("use_multithreading", True)
        if self.is_simulation:
            self.use_thread = False  # Luôn tắt threading trong simulation để dễ debug
        self.servo_name = "NeckLR"
        self.center_angle = float(config_dict.get("servo_center", 90.0))
        self.min_abs_angle = float(config_dict.get("servo_min_abs", 15.0))
        self.max_abs_angle = float(config_dict.get("servo_max_abs", 165.0))
        self.current_angle = self.center_angle
        self.target_angle = self.center_angle
        self.servo_direction = float(config_dict.get("servo_direction", -1.0))
        self.min_command_interval = float(config_dict.get("min_command_interval_sec", 0.08))
        self.servo_duration_ms = int(config_dict.get("servo_duration_ms", 80))
        self.debug_hardware = bool(config_dict.get("debug_hardware", False))
        self.debug_every_n = max(1, int(config_dict.get("debug_every_n_commands", 10)))
        self.last_command_time = 0.0
        self.command_count = 0
        self.last_response = None

        if not self.is_simulation:
            ip = config_dict.get("robot_ip", "127.0.0.1")
            try:
                init_resp = YanAPI.yan_api_init(ip)
                print("[HARDWARE] YanAPI init OK | ip={} | resp={}".format(ip, init_resp))
            except Exception as e:
                print("[ERROR] YanAPI init failed: {}".format(e))

        if self.use_thread:
            self.angle_queue = queue.Queue(maxsize=1)
            self.thread = threading.Thread(target=self._servo_worker)
            self.thread.daemon = True
            self.thread.start()
            print("[HARDWARE] Started | mode={} | threading=ON".format(
                "SIM" if is_simulation else "REAL"))
        else:
            print("[HARDWARE] Started | mode={} | threading=OFF".format(
                "SIM" if is_simulation else "REAL"))

    def _send_to_hardware(self, angle):
        prev = self.current_angle
        final_angle_int = int(round(angle))
        self.command_count += 1

        if self.is_simulation:
            self.current_angle = angle
            if abs(angle - prev) >= 0.1:  # nhạy hơn một chút để debug
                print("[SIM] NeckLR -> {}deg".format(final_angle_int))
        else:
            try:
                resp = YanAPI.set_servos_angles({self.servo_name: final_angle_int}, self.servo_duration_ms)
                self.last_response = resp
                self.current_angle = angle
                if self.debug_hardware and self.command_count % self.debug_every_n == 0:
                    print("[HARDWARE] set_servos_angles {}={} dur={} -> {}".format(
                        self.servo_name, final_angle_int, self.servo_duration_ms, resp))
            except Exception as e:
                print("[ERROR] YanAPI failed: {}".format(e))

    def get_current_angle(self):
        return self.current_angle

    def get_last_response(self):
        return self.last_response

    def set_head_angle(self, pid_correction):
        """
        Nhận giá trị hiệu chỉnh từ PID.
        Cập nhật góc hiện tại theo hướng incremental.
        """
        now = time.time()
        if now - self.last_command_time < self.min_command_interval:
            return
        self.last_command_time = now

        new_angle = self.target_angle + self.servo_direction * pid_correction
        
        # 2. Kẹp (Clamp) góc vào giới hạn an toàn của YanAPI (15 - 165)
        safe_angle = max(min(new_angle, self.max_abs_angle), self.min_abs_angle)
        self.target_angle = safe_angle
        
        # 3. Gửi lệnh đi
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
