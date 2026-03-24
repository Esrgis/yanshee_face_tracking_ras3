import cv2
import numpy as np
from ultralytics import YOLO

class VisionSystem:
    """
    Hệ thống Thị giác tích hợp YOLOv8/v12 và KCF Tracker.
    Hoạt động theo cơ chế Cascade: Ưu tiên Tracker (nhẹ) -> Fallback về YOLO (nặng).
    """
    def __init__(self, model_path="yolov8n-face.pt", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        print(f"[VISION INIT] Đang tải mô hình YOLO từ: {model_path} ...")
        
        try:
            # Tải mô hình YOLO (Dùng bản nano 'n' cho nhẹ)
            self.yolo = YOLO(model_path)
        except Exception as e:
            print(f"[VISION ERROR] LỖI CHÍ MẠNG: Không thể tải mô hình YOLO. Chi tiết: {e}")
            raise e
        
        self.tracker = None
        self.is_tracking = False
        self.bbox = None  # Lưu trữ tọa độ bounding box: (x, y, w, h)

    def _init_tracker(self, frame, bbox):
        """Khởi tạo lại KCF Tracker với bounding box mới từ YOLO"""
        # BẮT BUỘC dùng cv2.legacy vì KCF đã bị OpenCV chuyển sang module mở rộng
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.is_tracking = True
        self.bbox = bbox
        # print("[VISION SYSTEM] Đã khóa mục tiêu bằng KCF Tracker!")

    def _get_largest_face(self, results):
        """
        Lọc nhiễu đa mục tiêu: 
        Nếu có nhiều người trong khung hình, chỉ chọn khuôn mặt to nhất (gần robot nhất).
        Tránh hiện tượng robot bị phân tâm, giật cổ qua lại giữa 2 người.
        """
        best_box = None
        max_area = 0
        
        # YOLO trả về một list kết quả, lấy phần tử đầu tiên results[0]
        for box in results[0].boxes:
            if box.conf[0] >= self.conf_threshold:
                # Ép kiểu tọa độ về số nguyên
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                # So sánh diện tích để tìm mặt to nhất
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, w, h)
                    
        return best_box

    def process_frame(self, frame):
        """
        Hàm cốt lõi xử lý từng khung hình. Đẩy ảnh vào, lấy tọa độ tâm ra.
        
        :param frame: Ảnh numpy array đọc từ cv2.VideoCapture
        :return: (target_found, bbox, center_x, center_y)
        """
        target_found = False
        center_x, center_y = -1, -1

        # ---------------------------------------------------------
        # 1. GIAI ĐOẠN TRACKING (Ưu tiên số 1 vì nó nhẹ và nhanh)
        # ---------------------------------------------------------
        if self.is_tracking and self.tracker is not None:
            # Cập nhật Tracker
            success, box = self.tracker.update(frame)
            if success:
                self.bbox = tuple(map(int, box))
                target_found = True
            else:
                # KCF bị mất dấu (do vật cản, quay mặt đi, mờ nhòe...)
                self.is_tracking = False
                self.bbox = None
                # State Machine sẽ nhận cờ target_found = False ở frame này để chuyển trạng thái!

        # ---------------------------------------------------------
        # 2. GIAI ĐOẠN DETECTION (Kích hoạt khi chưa Tracking hoặc vừa Mất dấu)
        # ---------------------------------------------------------
        if not self.is_tracking:
            # verbose=False để giấu cái log lộn xộn của Ultralytics mỗi khi predict
            results = self.yolo.predict(frame, verbose=False)
            best_box = self._get_largest_face(results)
            
            if best_box is not None:
                # YOLO tìm thấy mặt -> Bàn giao tọa độ cho KCF để khóa mục tiêu
                self._init_tracker(frame, best_box)
                target_found = True

        # ---------------------------------------------------------
        # 3. TÍNH TOÁN TÂM MỤC TIÊU (Dữ liệu đầu ra cho Kalman/PID)
        # ---------------------------------------------------------
        if target_found and self.bbox is not None:
            x, y, w, h = self.bbox
            center_x = x + w // 2
            center_y = y + h // 2

        return target_found, self.bbox, center_x, center_y