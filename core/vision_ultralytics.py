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

    def _get_largest_face(self, results):
        best_box = None
        max_area = 0
        for box in results[0].boxes:
            if box.conf[0] >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, w, h)
        return best_box

    def process_frame(self, frame):
        target_found = False
        center_x, center_y = -1, -1

        if self.is_tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                self.bbox = tuple(map(int, box))
                target_found = True
            else:
                self.is_tracking = False
                self.bbox = None

        if not self.is_tracking:
            results = self.yolo.predict(frame, verbose=False)
            best_box = self._get_largest_face(results)
            if best_box is not None:
                self._init_tracker(frame, best_box)
                target_found = True

        if target_found and self.bbox is not None:
            x, y, w, h = self.bbox
            center_x = x + w // 2
            center_y = y + h // 2

        return target_found, self.bbox, center_x, center_y