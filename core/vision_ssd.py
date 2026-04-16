#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vision_ssd.py -- SSD MobileNet face detector + KCF tracker
Dung cho Buoc 1 benchmark: chung minh deep learning qua nang cho Pi 3B.

Can 2 file model (dat vao thu muc goc repo):
  opencv_face_detector.pbtxt
  opencv_face_detector_uint8.pb

Download:
  https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb
"""
import cv2
import os
from core.vision import VisionSystem


class VisionSSD(VisionSystem):

    def __init__(self, prototxt_path=None, model_path=None,
                 conf_threshold=0.5, detection_skip=5,
                 pad_ratio=0.20, iou_reinit_threshold=0.5,
                 max_jump_px=180):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if prototxt_path is None:
            prototxt_path = os.path.join(base_dir, "opencv_face_detector.pbtxt")
        if model_path is None:
            model_path = os.path.join(base_dir, "opencv_face_detector_uint8.pb")

        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            raise Exception(
                "SSD model files not found. Download va dat vao thu muc goc repo.\n"
                "Xem docstring o dau file."
            )

        self.net              = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
        self.conf_threshold   = float(conf_threshold)
        self.detection_skip   = max(1, detection_skip)
        self.pad_ratio        = float(pad_ratio)
        self.iou_reinit_threshold = float(iou_reinit_threshold)
        self.max_jump_px      = int(max_jump_px)

        self.frame_counter = 0
        self.tracker       = None
        self.is_tracking   = False
        self.bbox          = None
        self.last_center   = None

        print("[SSD-KCF] Init OK | conf={} | skip={} | pad={:.0%}".format(
            self.conf_threshold, self.detection_skip, self.pad_ratio))

    def process_frame(self, frame, prev_x=-1, prev_y=-1):
        target_found = False
        center_x = center_y = -1
        self.frame_counter += 1
        fh, fw = frame.shape[:2]

        # --- 1. Cap nhat KCF ---
        if self.is_tracking and self.tracker is not None:
            try:
                ok, box = self.tracker.update(frame)
                if ok and box[2] > 0 and box[3] > 0:
                    self.bbox    = tuple(map(int, box))
                    target_found = True
                else:
                    self._reset_tracker()
            except Exception:
                self._reset_tracker()

        # --- 2. Detection ---
        run_detection = (not self.is_tracking or not target_found
                         or self.frame_counter >= self.detection_skip)

        if run_detection:
            if self.frame_counter >= self.detection_skip:
                self.frame_counter = 0

            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            detections = self.net.forward()  # shape: (1,1,N,7)

            best_raw = self._select_best_face(detections, fw, fh)

            if best_raw is not None:
                best_padded   = self._add_padding(best_raw, frame.shape)
                should_reinit = (
                    not self.is_tracking
                    or self.bbox is None
                    or self._iou(self.bbox, best_padded) < self.iou_reinit_threshold
                )
                if should_reinit:
                    self._init_tracker(frame, best_padded)
                self.bbox    = best_padded if self.bbox is None else self.bbox
                target_found = True
            else:
                if not self.is_tracking:
                    target_found = False
                    self.bbox    = None

        # --- 3. Center ---
        if target_found and self.bbox is not None:
            x, y, w, h       = self.bbox
            center_x          = x + w // 2
            center_y          = y + h // 2
            self.last_center  = (center_x, center_y)
        else:
            self.bbox = None

        return target_found, self.bbox, center_x, center_y

    def _select_best_face(self, detections, fw, fh):
        """
        Parse SSD output, loc theo conf_threshold va max_jump_px.
        Tra ve bbox (x,y,w,h) tot nhat hoac None.
        """
        candidates = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * fw)
            y1 = int(detections[0, 0, i, 4] * fh)
            x2 = int(detections[0, 0, i, 5] * fw)
            y2 = int(detections[0, 0, i, 6] * fh)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(fw, x2); y2 = min(fh, y2)
            w  = x2 - x1;     h  = y2 - y1
            if w <= 0 or h <= 0:
                continue
            candidates.append((x1, y1, w, h, conf))

        if not candidates:
            return None

        if self.last_center is None:
            return max(candidates, key=lambda b: b[4])[:4]  # conf cao nhat

        cx_prev, cy_prev = self.last_center

        def score(b):
            x, y, w, h, conf = b
            dist = ((x + w//2 - cx_prev)**2 + (y + h//2 - cy_prev)**2) ** 0.5
            if dist > self.max_jump_px:
                return -1.0
            return (w * h * conf) / (dist + 1.0)

        valid = [c for c in candidates if score(c) > 0]
        if not valid:
            return None
        best = max(valid, key=score)
        return (best[0], best[1], best[2], best[3])

    def _add_padding(self, bbox, frame_shape):
        x, y, w, h = bbox
        fh, fw     = frame_shape[:2]
        px = int(w * self.pad_ratio)
        py = int(h * self.pad_ratio)
        x2 = max(0, x - px)
        y2 = max(0, y - py)
        return (x2, y2, min(fw - x2, w + 2*px), min(fh - y2, h + 2*py))

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix    = max(0, min(ax+aw, bx+bw) - max(ax, bx))
        iy    = max(0, min(ay+ah, by+bh) - max(ay, by))
        inter = ix * iy
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    def _init_tracker(self, frame, bbox):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False
        try:
            self.tracker = cv2.TrackerKCF_create()
            ok = self.tracker.init(frame, (x, y, w, h))
            if ok:
                self.is_tracking = True
                self.bbox        = (x, y, w, h)
                return True
            self._reset_tracker()
            return False
        except Exception as e:
            print("[KCF] Init error:", e)
            self._reset_tracker()
            return False

    def _reset_tracker(self):
        self.tracker     = None
        self.is_tracking = False
        self.bbox        = None