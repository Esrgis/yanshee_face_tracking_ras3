import cv2
import numpy as np
from core.vision import VisionSystem

IMGSZ = 640

class VisionONNX(VisionSystem):
    def __init__(self, model_path="models/yolov8n-face-lindevs.onnx", conf_threshold=0.5):
        self.conf_threshold = float(conf_threshold)
        self.imgsz = IMGSZ
        print(f"[VISION INIT] Backend: ONNX | Model: {model_path}")
        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[VISION INIT] Tải model ONNX thành công!")
        except Exception as e:
            print(f"[VISION ERROR] Không thể tải ONNX model: {e}")
            raise e
        self.tracker = None
        self.is_tracking = False
        self.bbox = None

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = self.imgsz / max(h, w)
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas[0:new_h, 0:new_w] = resized
        return canvas, scale

    def _parse_detections(self, output, scale, orig_w, orig_h):
        detections = output[0].T
        boxes = []
        confidences = []
        for row in detections:
            cx, cy, bw, bh, conf = row
            if conf < self.conf_threshold:
                continue
            x1 = int((cx - bw / 2) * scale)
            y1 = int((cy - bh / 2) * scale)
            w  = int(bw * scale)
            h  = int(bh * scale)
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            w  = max(1, min(w,  orig_w - x1))
            h  = max(1, min(h,  orig_h - y1))
            boxes.append([x1, y1, w, h])
            confidences.append(float(conf))

        if not boxes:
            return None

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.45)
        if len(indices) == 0:
            return None

        best_idx = None
        best_conf = -1
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            if confidences[idx] > best_conf:
                best_conf = confidences[idx]
                best_idx = idx

        return tuple(boxes[best_idx])

    def _init_tracker(self, frame, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.is_tracking = True
        self.bbox = bbox

    def process_frame(self, frame):
        target_found = False
        center_x, center_y = -1, -1
        orig_h, orig_w = frame.shape[:2]

        if self.is_tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                self.bbox = tuple(map(int, box))
                target_found = True
            else:
                self.is_tracking = False
                self.bbox = None

        if not self.is_tracking:
            letterboxed, scale = self._letterbox(frame)
            blob = cv2.dnn.blobFromImage(
                letterboxed,
                scalefactor=1.0 / 255.0,
                size=(self.imgsz, self.imgsz),
                swapRB=True,
                crop=False
            )
            self.net.setInput(blob)
            output = self.net.forward()
            inv_scale = 1.0 / scale
            best_box = self._parse_detections(output, inv_scale, orig_w, orig_h)
            if best_box is not None:
                self._init_tracker(frame, best_box)
                target_found = True

        if target_found and self.bbox is not None:
            x, y, w, h = self.bbox
            center_x = x + w // 2
            center_y = y + h // 2

        return target_found, self.bbox, center_x, center_y