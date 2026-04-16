#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vision_lbp.py -- LBP Cascade face detector + KCF tracker
Same interface as VisionHaarCascade, drop-in replacement for benchmark.

LBP vs Haar:
  - LBP nhanh hon ~2x (integer ops thay float)
  - Accuracy thap hon Haar mot chut, nhat la voi mat nghieng
  - Dung de so sanh toc do vs Haar trong Buoc 1 benchmark
"""
import cv2
import os
from core.vision import VisionSystem


class VisionLBP(VisionSystem):

    def __init__(self, cascade_path=None, conf_threshold=0.5,
                 detection_skip=5, pad_ratio=0.20, iou_reinit_threshold=0.5,
                 max_jump_px=180):
        """
        Parameters
        ----------
        cascade_path : str
            Duong dan toi lbpcascade_frontalface_improved.xml.
            Mac dinh tim trong thu muc goc repo.
        pad_ratio : float
            Padding them vao bbox truoc khi init KCF. Default 0.20.
        iou_reinit_threshold : float
            Chi reinit KCF khi IoU(bbox_cu, bbox_moi) < threshold.
        max_jump_px : int
            Loai candidate neu tam cach vi tri truoc > gia tri nay.
        """
        if cascade_path is None:
            # Tim trong thu muc goc repo (canh file main)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(base_dir, "lbpcascade_frontalface_improved.xml")
            if os.path.exists(default_path):
                cascade_path = default_path
            else:
                raise Exception(
                    "LBP cascade not found. Download from:\n"
                    "https://raw.githubusercontent.com/opencv/opencv/master/"
                    "data/lbpcascades/lbpcascade_frontalface_improved.xml\n"
                    "Va dat vao thu muc goc repo."
                )

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise IOError("Cannot load LBP cascade from " + cascade_path)

        self.detection_skip       = max(1, detection_skip)
        self.pad_ratio            = float(pad_ratio)
        self.iou_reinit_threshold = float(iou_reinit_threshold)
        self.max_jump_px          = int(max_jump_px)

        self.frame_counter = 0
        self.tracker       = None
        self.is_tracking   = False
        self.bbox          = None
        self.last_center   = None

        print("[LBP-KCF] Init OK | skip={} | pad={:.0%} | iou_thr={}".format(
            self.detection_skip, self.pad_ratio, self.iou_reinit_threshold))

    # ------------------------------------------------------------------
    # Public API -- giu nguyen interface voi phan con lai cua pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame, prev_x=-1, prev_y=-1):
        """
        Returns: (target_found, bbox, center_x, center_y)
        bbox = (x, y, w, h) co padding
        """
        target_found = False
        center_x = center_y = -1
        self.frame_counter += 1

        # --- 1. Cap nhat KCF tracker ---
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

        # --- 2. Quyet dinh co chay detection khong ---
        run_detection = (not self.is_tracking or not target_found
                         or self.frame_counter >= self.detection_skip)

        if run_detection:
            if self.frame_counter >= self.detection_skip:
                self.frame_counter = 0

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # LBP dung scaleFactor nho hon Haar (1.05 thay 1.08)
            # vi LBP it sensitive hon voi scale -- can nhieu buoc hon
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3,
                minSize=(40, 40), maxSize=(400, 400)
            )

            if len(faces) > 0:
                best_raw = self._select_best_face(faces)

                if best_raw is not None:
                    best_padded  = self._add_padding(best_raw, frame.shape)
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

        # --- 3. Tinh center ---
        if target_found and self.bbox is not None:
            x, y, w, h   = self.bbox
            center_x      = x + w // 2
            center_y      = y + h // 2
            self.last_center = (center_x, center_y)
        else:
            self.bbox = None

        return target_found, self.bbox, center_x, center_y

    # ------------------------------------------------------------------
    # Private helpers -- copy y chang Haar de dam bao nhat quan
    # ------------------------------------------------------------------

    def _add_padding(self, bbox, frame_shape):
        x, y, w, h = bbox
        fh, fw     = frame_shape[:2]
        px         = int(w * self.pad_ratio)
        py         = int(h * self.pad_ratio)
        x2         = max(0,      x - px)
        y2         = max(0,      y - py)
        w2         = min(fw - x2, w + 2 * px)
        h2         = min(fh - y2, h + 2 * py)
        return (x2, y2, w2, h2)

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix    = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy    = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _select_best_face(self, faces):
        if self.last_center is None:
            return max(faces, key=lambda b: b[2] * b[3])

        cx_prev, cy_prev = self.last_center

        def score(b):
            x, y, w, h = b
            bcx  = x + w // 2
            bcy  = y + h // 2
            dist = ((bcx - cx_prev) ** 2 + (bcy - cy_prev) ** 2) ** 0.5
            if dist > self.max_jump_px:
                return -1.0
            return (w * h) / (dist + 1.0)

        candidates = [f for f in faces if score(f) > 0]
        if not candidates:
            return None
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