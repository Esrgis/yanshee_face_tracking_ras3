import cv2
from ultralytics import YOLO
from core.vision import VisionSystem

class VisionUltralytics(VisionSystem):
    def __init__(self, model_path="models/yolov12n-face.pt", conf_threshold=0.5):
        self.conf_threshold = float(conf_threshold)
        print(f"[VISION INIT] Backend: Ultralytics | Model: {model_path}")
        try:
            self.yolo = YOLO(model_path)
        except Exception as e:
            print(f"[VISION ERROR] Không thể tải model .pt: {e}")
            raise e
        self.tracker = None
        self.is_tracking = False
        self.bbox = None

    def _init_tracker(self, frame, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.is_tracking = True
        self.bbox = bbox

    def _get_best_face(self, results, prev_x, prev_y):
        """
        Lọc nhiễu thông minh bằng 'Neo Không Gian'.
        Nếu chưa có mục tiêu (prev_x == -1), chọn mặt to nhất.
        Nếu ĐÃ CÓ mục tiêu bị mất dấu, tìm khuôn mặt nằm GẦN vị trí cũ nhất!
        """
        best_box = None
        
        # TRƯỜNG HỢP 1: Bắt đầu video, chưa từng track ai -> Ưu tiên mặt to nhất
        if prev_x == -1 or prev_y == -1:
            max_area = 0
            for box in results[0].boxes:
                if box.conf[0] >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2 - x1, y2 - y1)
            return best_box

        # TRƯỜNG HỢP 2: Tìm lại mục tiêu vừa mất -> Tìm thằng gần nhất với vị trí cũ
        min_distance = float('inf')
        for box in results[0].boxes:
            if box.conf[0] >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                center_x = x1 + w // 2
                center_y = y1 + h // 2
                
                # Tính khoảng cách Euclid từ tâm khuôn mặt này đến vị trí cũ
                distance = math.hypot(center_x - prev_x, center_y - prev_y)
                
                if distance < min_distance:
                    min_distance = distance
                    best_box = (x1, y1, w, h)
                    
        return best_box

    def process_frame(self, frame, prev_x=-1, prev_y=-1):
        """
        Anh phải sửa lại hàm process_frame để nó nhận thêm tọa độ quá khứ
        """
        target_found = False
        center_x, center_y = -1, -1

        # 1. KCF Tracking
        if self.is_tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                self.bbox = tuple(map(int, box))
                target_found = True
            else:
                self.is_tracking = False
                self.bbox = None

        # 2. YOLO Fallback (Có Neo không gian)
        if not self.is_tracking:
            results = self.yolo.predict(frame, verbose=False)
            
            # ĐƯA TỌA ĐỘ QUÁ KHỨ VÀO ĐÂY:
            best_box = self._get_best_face(results, prev_x, prev_y)
            
            if best_box is not None:
                self._init_tracker(frame, best_box)
                target_found = True

        # 3. Tính toán lại tâm... (Giữ nguyên như cũ)
        if target_found and self.bbox is not None:
            x, y, w, h = self.bbox
            center_x = x + w // 2
            center_y = y + h // 2

        return target_found, self.bbox, center_x, center_y