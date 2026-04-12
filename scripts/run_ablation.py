#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_ablation.py - Ablation study runner (Python 3.5 compatible)

Chay:
  python scripts/run_ablation.py --source 0 --duration 60
  make ablation
  make ablation SRC=data/videos/test.avi DURATION=90 CONFIGS=AB

4 configs:
  A - Static skip=1  (baseline nang)
  B - Static skip=5  (baseline nhe)
  C - Adaptive, velocity only
  D - Adaptive, velocity + jitter  (proposed)

Output: results/logs/ablation_*.csv
"""

from __future__ import print_function
import cv2
import csv
import time
import os
import argparse
import collections
import numpy as np


class TrackerKalmanFilter(object):
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)

    def init_state(self, x, y):
        self.kf.statePre  = np.array([[np.float32(x)],[np.float32(y)],[0],[0]], np.float32)
        self.kf.statePost = np.array([[np.float32(x)],[np.float32(y)],[0],[0]], np.float32)

    def predict(self):
        p = self.kf.predict()
        return int(p[0,0]), int(p[1,0])

    def update(self, x, y):
        self.kf.correct(np.array([[np.float32(x)],[np.float32(y)]]))
        s = self.kf.statePost
        return int(s[0,0]), int(s[1,0])


class PIDController(object):
    def __init__(self, Kp=0.05, Ki=0.0, Kd=0.01,
                 max_integral=50.0, deadzone=2.0, output_limit=15.0):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd
        self.max_integral=max_integral
        self.deadzone=deadzone
        self.output_limit=output_limit
        self.prev_error=0.0; self.integral=0.0

    def update(self, error, dt):
        if dt <= 0: return 0.0
        if abs(error) < self.deadzone: error = 0.0
        P = self.Kp * error
        self.integral = max(-self.max_integral,
                            min(self.max_integral, self.integral + error * dt))
        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error
        out = P + self.Ki * self.integral + D
        return max(-self.output_limit, min(self.output_limit, out))


class AdaptiveScheduler(object):
    def __init__(self, enabled=True, base_skip=5, min_skip=1,
                 max_skip=15, alpha=0.08, beta=0.05, window=5):
        self.enabled   = enabled
        self.base_skip = int(base_skip)
        self.min_skip  = int(min_skip)
        self.max_skip  = int(max_skip)
        self.alpha     = float(alpha)
        self.beta      = float(beta)
        self._cx = collections.deque(maxlen=window)
        self._cy = collections.deque(maxlen=window)
        self.current_skip  = base_skip
        self.last_velocity = 0.0
        self.last_reason   = "init"

    def compute(self, cx, cy, jitter, lost):
        if not self.enabled:
            self.current_skip = self.base_skip
            self.last_reason  = "disabled"
            return self.current_skip
        if lost:
            self.current_skip = self.min_skip
            self.last_reason  = "lost"
            self._upd(cx, cy)
            return self.current_skip
        vel   = self._vel(cx, cy)
        self._upd(cx, cy)
        skip  = int(round(self.base_skip / (1.0 + self.alpha*vel + self.beta*jitter)))
        skip  = max(self.min_skip, min(self.max_skip, skip))
        self.last_velocity = round(vel, 2)
        self.last_reason   = "adaptive"
        self.current_skip  = skip
        return skip

    def _vel(self, cx, cy):
        if cx < 0 or len(self._cx) < 2: return 0.0
        n  = len(self._cx)
        dx = (cx - self._cx[0]) / n
        dy = (cy - self._cy[0]) / n
        return (dx*dx + dy*dy) ** 0.5

    def _upd(self, cx, cy):
        if cx >= 0:
            self._cx.append(cx)
            self._cy.append(cy)


class VisionHaarKCF(object):
    def __init__(self, cascade_path, skip=5, pad=0.20, iou_thr=0.5, max_jump=180):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise IOError("Cannot load cascade: " + cascade_path)
        self.skip        = skip
        self.pad         = pad
        self.iou_thr     = iou_thr
        self.max_jump    = max_jump
        self.frame_cnt   = 0
        self.tracker     = None
        self.is_tracking = False
        self.bbox        = None
        self.last_center = None

    def process(self, frame):
        found = False; cx = cy = -1
        self.frame_cnt += 1
        if self.is_tracking and self.tracker:
            try:
                ok, box = self.tracker.update(frame)
                if ok and box[2] > 0 and box[3] > 0:
                    self.bbox = tuple(map(int, box)); found = True
                else:
                    self._rst()
            except Exception:
                self._rst()
        if not self.is_tracking or not found or self.frame_cnt >= self.skip:
            if self.frame_cnt >= self.skip: self.frame_cnt = 0
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, 1.08, 5, minSize=(40,40), maxSize=(400,400))
            if len(faces) > 0:
                best = self._best(faces)
                if best is not None:
                    pad = self._pad(best, frame.shape)
                    if not self.is_tracking or self.bbox is None or                        self._iou(self.bbox, pad) < self.iou_thr:
                        self._init(frame, pad)
                    if self.bbox is None: self.bbox = pad
                    found = True
            else:
                if not self.is_tracking:
                    found = False; self.bbox = None
        if found and self.bbox:
            x,y,w,h = self.bbox
            cx = x + w//2; cy = y + h//2
            self.last_center = (cx, cy)
        else:
            self.bbox = None
        return found, self.bbox, cx, cy

    def _pad(self, b, shape):
        x,y,w,h = b; fh,fw = shape[:2]
        px = int(w*self.pad); py = int(h*self.pad)
        x2 = max(0, x-px); y2 = max(0, y-py)
        return (x2, y2, min(fw-x2, w+2*px), min(fh-y2, h+2*py))

    def _iou(self, a, b):
        ax,ay,aw,ah = a; bx,by,bw,bh = b
        ix = max(0, min(ax+aw,bx+bw)-max(ax,bx))
        iy = max(0, min(ay+ah,by+bh)-max(ay,by))
        i  = ix*iy; u = aw*ah+bw*bh-i
        return i/u if u>0 else 0.0

    def _best(self, faces):
        if self.last_center is None:
            return max(faces, key=lambda b: b[2]*b[3])
        pcx,pcy = self.last_center
        def sc(b):
            x,y,w,h = b
            d = ((x+w//2-pcx)**2+(y+h//2-pcy)**2)**0.5
            return -1.0 if d>self.max_jump else (w*h)/(d+1.0)
        cands = [f for f in faces if sc(f) > 0]
        return max(cands, key=sc) if cands else None

    def _init(self, frame, bbox):
        x,y,w,h = bbox
        if w<=0 or h<=0: return
        try:
            self.tracker = cv2.TrackerKCF_create()
            if self.tracker.init(frame, (x,y,w,h)):
                self.is_tracking = True; self.bbox = (x,y,w,h)
            else:
                self._rst()
        except Exception:
            self._rst()

    def _rst(self):
        self.tracker = None; self.is_tracking = False; self.bbox = None


CONFIGS = [
    {"name": "A_static_skip1",      "enabled": False, "base_skip": 1,
     "alpha": 0.0,  "beta": 0.0},
    {"name": "B_static_skip5",      "enabled": False, "base_skip": 5,
     "alpha": 0.0,  "beta": 0.0},
    {"name": "C_adaptive_vel_only", "enabled": True,  "base_skip": 5,
     "alpha": 0.08, "beta": 0.0},
    {"name": "D_adaptive_full",     "enabled": True,  "base_skip": 5,
     "alpha": 0.08, "beta": 0.05},
]

CASCADE = "haarcascade_frontalface_default.xml"
W, H    = 640, 480
LOG_DIR = "results/logs"


def run_one(cfg, source, duration, cascade):
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    out_csv = os.path.join(LOG_DIR, "ablation_{}.csv".format(cfg["name"]))

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc: " + str(source)); return

    vision = VisionHaarKCF(cascade, skip=cfg["base_skip"])
    kalman = TrackerKalmanFilter()
    pid    = PIDController()
    sched  = AdaptiveScheduler(
        enabled   = cfg["enabled"],
        base_skip = cfg["base_skip"],
        alpha     = cfg["alpha"],
        beta      = cfg["beta"],
    )
    k_init = False; n = 0
    t0 = prev = time.time()

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame","found","cx_raw","cx_filtered","jitter",
                     "error_px","pid_output","fps","vision_ms",
                     "sched_skip","sched_velocity","sched_reason","config_name"])
        print("[{}] running...".format(cfg["name"]))

        while True:
            now = time.time()
            if duration and (now - t0) >= duration: break
            dt   = now - prev; prev = now
            ret, frame = cap.read()
            if not ret:
                if not isinstance(source, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            frame = cv2.resize(frame, (W, H)); n += 1
            tv = time.time()
            found, bbox, cx_raw, cy_raw = vision.process(frame)
            vis_ms = (time.time() - tv) * 1000

            jit = 0.0; cx_f = cx_raw
            if found and cx_raw >= 0:
                if not k_init:
                    kalman.init_state(cx_raw, cy_raw); k_init = True
                else:
                    cx_f, _ = kalman.update(cx_raw, cy_raw)
                jit = abs(cx_raw - cx_f)
            elif k_init:
                cx_f, _ = kalman.predict()

            sk         = sched.compute(cx_f, cy_raw, jit, not found)
            vision.skip = sk
            err        = float(cx_f - W//2) if found else 0.0
            pid_out    = pid.update(err, dt if dt > 0 else 0.033)
            fps        = 1.0/dt if dt > 0 else 0.0

            wr.writerow([n, int(found), cx_raw, cx_f,
                         round(jit,2), round(err,2), round(pid_out,3),
                         round(fps,2), round(vis_ms,2),
                         sk, sched.last_velocity,
                         sched.last_reason, cfg["name"]])

            if n % 30 == 0:
                print("  f={:4d} fps={:5.1f} found={} skip={}".format(
                    n, fps, found, sk))

    cap.release()
    print("  -> " + out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",   default="0")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--cascade",  default=CASCADE)
    ap.add_argument("--configs",  default="ABCD")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    mp  = {"A":0,"B":1,"C":2,"D":3}
    sel = [CONFIGS[mp[c]] for c in args.configs.upper() if c in mp]

    print("="*50)
    print(" ABLATION: {} config(s), {}s each".format(len(sel), args.duration))
    print("="*50)
    for cfg in sel:
        run_one(cfg, src, args.duration, args.cascade)
    print("\nDone. Tiep theo: make analyze")


if __name__ == "__main__":
    main()
