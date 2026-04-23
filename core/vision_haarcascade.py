#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vision_haarcascade.py  —  Fixed version
Thay đổi so với bản gốc:
  1. _add_padding()       : thêm margin 20% quanh bbox trước khi init KCF
                            → KCF không mất feature khi mặt xoay nhẹ
  2. _iou()               : kiểm tra overlap giữa bbox cũ và bbox mới
                            → không reinit KCF khi Haar trả bbox lệch nhỏ
  3. _select_best_face()  : chọn mặt gần vị trí trước nhất (không chỉ lớn nhất)
                            → tránh nhảy sang track quạt / vật thể nền
"""
import cv2
import os
from core.vision import VisionSystem


class VisionHaarCascade(VisionSystem):

    def __init__(self, cascade_path=None, conf_threshold=0.5,
             detection_skip=5, pad_ratio=0.20, iou_reinit_threshold=0.5,
             max_jump_px=180, min_size=30, max_size=400,
             min_neighbors=5, scale_factor=1.08):
        """
        Parameters
        ----------
        pad_ratio : float
            Tỉ lệ padding thêm vào bbox Haar trước khi init KCF.
            0.20 = thêm 20% mỗi phía. Giúp KCF bao trọn mặt khi xoay.
        iou_reinit_threshold : float
            Chỉ reinit KCF khi IoU(bbox_cũ, bbox_mới) < threshold này.
            Tránh bbox nhảy khi Haar và KCF nhất quán.
        max_jump_px : int
            Loại bỏ detection candidate nếu tâm cách vị trí trước > giá trị này.
            Hàng rào chặn quạt / vật thể ngẫu nhiên pass Haar.
        """
        # --- Tìm cascade path ---
        if cascade_path is None:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            else:
                cv2_dir = os.path.dirname(cv2.__file__)
                possible = os.path.join(cv2_dir, 'data', 'haarcascade_frontalface_default.xml')
                if os.path.exists(possible):
                    cascade_path = possible
                else:
                    raise Exception("Cascade not found. Set cascade_path in config.json")

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise IOError("Cannot load cascade from " + cascade_path)

        self.detection_skip       = max(1, detection_skip)
        self.pad_ratio            = float(pad_ratio)
        self.iou_reinit_threshold = float(iou_reinit_threshold)
        self.max_jump_px          = int(max_jump_px)
        self.min_size      = (int(min_size), int(min_size))
        self.max_size      = (int(max_size), int(max_size))
        self.min_neighbors = int(min_neighbors)
        self.scale_factor  = float(scale_factor)

        self.frame_counter = 0
        self.tracker       = None
        self.is_tracking   = False
        self.bbox          = None                 # bbox hiện tại (đã padded)
        self.last_center   = None                 # (cx, cy) frame trước

        print("[Haar-KCF] Init OK | skip={} | pad={:.0%} | iou_thr={} | minN={} | minSz={}".format(
        self.detection_skip, self.pad_ratio, self.iou_reinit_threshold,
        self.min_neighbors, self.min_size))

    # ------------------------------------------------------------------
    # Public API (giữ nguyên interface với phần còn lại của pipeline)
    # ------------------------------------------------------------------

    def process_frame(self, frame, prev_x=-1, prev_y=-1):
        """
        Returns: (target_found, bbox, center_x, center_y)
        bbox = (x, y, w, h) — có padding, dùng để vẽ và init KCF
        """
        target_found = False
        center_x = center_y = -1
        self.frame_counter += 1

        # --- 1. Cập nhật KCF tracker ---
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

        # --- 2. Quyết định có chạy detection không ---
        run_detection = (not self.is_tracking or not target_found
                         or self.frame_counter >= self.detection_skip)

        if run_detection:
            if self.frame_counter >= self.detection_skip:
                self.frame_counter = 0

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor  = self.scale_factor,
                minNeighbors = self.min_neighbors,
                minSize      = self.min_size,
                maxSize      = self.max_size,
            )

            if len(faces) > 0:
                # FIX 3: chọn mặt gần vị trí trước, không chỉ lớn nhất
                best_raw = self._select_best_face(faces)

                if best_raw is not None:
                    # FIX 1: thêm padding trước khi init KCF
                    best_padded = self._add_padding(best_raw, frame.shape)

                    # FIX 2: chỉ reinit KCF khi bbox thay đổi đáng kể
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
                # Haar không thấy gì
                if not self.is_tracking:
                    target_found = False
                    self.bbox    = None

        # --- 3. Tính center ---
        if target_found and self.bbox is not None:
            x, y, w, h  = self.bbox
            center_x     = x + w // 2
            center_y     = y + h // 2
            self.last_center = (center_x, center_y)
        else:
            self.bbox = None

        return target_found, self.bbox, center_x, center_y

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_padding(self, bbox, frame_shape):
        """FIX 1: Thêm padding quanh bbox, clamp trong frame."""
        x, y, w, h    = bbox
        fh, fw        = frame_shape[:2]
        px            = int(w * self.pad_ratio)
        py            = int(h * self.pad_ratio)
        x2            = max(0,      x - px)
        y2            = max(0,      y - py)
        w2            = min(fw - x2, w + 2 * px)
        h2            = min(fh - y2, h + 2 * py)
        return (x2, y2, w2, h2)

    def _iou(self, a, b):
        """FIX 2: IoU giữa hai bbox (x,y,w,h). Trả 0.0 nếu không overlap."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _select_best_face(self, faces):
        """
        FIX 3: Chọn face candidate tốt nhất.
        - Nếu chưa có lịch sử: chọn lớn nhất (hành vi cũ).
        - Nếu có last_center: chọn face gần nhất mà không nhảy quá max_jump_px.
          Score = area / (dist + 1) — ưu tiên lớn VÀ gần.
        """
        if self.last_center is None:
            return max(faces, key=lambda b: b[2] * b[3])

        cx_prev, cy_prev = self.last_center

        def score(b):
            x, y, w, h = b
            bcx = x + w // 2
            bcy = y + h // 2
            dist = ((bcx - cx_prev) ** 2 + (bcy - cy_prev) ** 2) ** 0.5
            if dist > self.max_jump_px:
                return -1.0           # loại — nhảy quá xa, có thể là quạt
            return (w * h) / (dist + 1.0)

        candidates = [f for f in faces if score(f) > 0]
        if not candidates:
            return None               # tất cả đều nhảy quá xa → không update
        return max(candidates, key=score)

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
            else:
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