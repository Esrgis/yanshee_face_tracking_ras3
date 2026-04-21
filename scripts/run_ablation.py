#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_ablation.py -- Buoc 2: Danh gia Scheduler & Tracking Trade-off (Pareto Frontier)

Chay:
  python scripts/run_ablation.py --videos data/videos --duration 60
"""
from __future__ import print_function
import cv2
import csv
import time
import os
import argparse
import math
from core.filters import TrackerKalmanFilter

W, H = 320, 240
LOG_DIR = "results/logs"

# ------------------------------------------------------------------
# 1. CORE INJECTION (STRICT GUARDRAIL: DO NOT MODIFY CORE FOLDER)
# ------------------------------------------------------------------
from core.vision_haarcascade import VisionHaarCascade

class VisionHaarMOSSE(VisionHaarCascade):
    """
    Subclass tiêm MOSSE Tracker trực tiếp tại Runtime. 
    Không làm thay đổi mã nguồn gốc của hệ thống.
    """
    def __init__(self, detection_skip=1):
        super(VisionHaarMOSSE, self).__init__(detection_skip=detection_skip)
        self.mosse_fallback = False

    def _init_tracker(self, frame, bbox):
        # Override hàm khởi tạo tracker của class cha
        try:
            self.tracker = cv2.TrackerMOSSE_create()
            self.mosse_fallback = False
        except AttributeError:
            self.tracker = cv2.legacy.TrackerMOSSE_create()
            self.mosse_fallback = True
        
        if bbox is not None:
            self.tracker.init(frame, bbox)
            self.is_tracking = True
        else:
            self.is_tracking = False

# ------------------------------------------------------------------
# 2. FILTER & SCHEDULER IMPLEMENTATION
# ------------------------------------------------------------------
class TrackerKalmanFilter:
    def __init__(self, gain=0.8):
        self.gain = gain
        self.last_val = None

    def update(self, current_val):
        if self.last_val is None:
            self.last_val = current_val
            return current_val
        filtered = self.gain * current_val + (1 - self.gain) * self.last_val
        self.last_val = filtered
        return filtered

    def reset(self):
        self.last_val = None

class AdaptiveScheduler:
    def __init__(self, config):
        self.is_adaptive = config['adaptive']
        self.base_skip   = config['base_skip']
        self.alpha       = config['alpha']
        self.beta        = config['beta']
        self.current_skip = self.base_skip

    def step(self, velocity, jitter):
        if not self.is_adaptive:
            return self.base_skip, "static"
        
        denominator = 1.0 + self.alpha * velocity + self.beta * jitter
        new_skip = int(round(self.base_skip / denominator))
        new_skip = max(1, min(self.base_skip, new_skip))
        reason = "adaptive" if new_skip < self.base_skip else "base"
        self.current_skip = new_skip
        return new_skip, reason

# ------------------------------------------------------------------
# 3. ABLATION CONFIGURATIONS (FROM SPEC)
# ------------------------------------------------------------------
CONFIGS = {
    "A": {"name": "A_static_skip1", "adaptive": False, "base_skip": 1, "alpha": 0.0, "beta": 0.0, "tracker": "kcf"},
    "B": {"name": "B_static_skip5", "adaptive": False, "base_skip": 5, "alpha": 0.0, "beta": 0.0, "tracker": "kcf"},
    "C": {"name": "C_adaptive_vel_only", "adaptive": True, "base_skip": 5, "alpha": 0.08, "beta": 0.0, "tracker": "kcf"},
    "D": {"name": "D_adaptive_full", "adaptive": True, "base_skip": 5, "alpha": 0.08, "beta": 0.05, "tracker": "kcf"},
    "E": {"name": "E_static_skip5_mosse", "adaptive": False, "base_skip": 5, "alpha": 0.0, "beta": 0.0, "tracker": "mosse"}
}

# ------------------------------------------------------------------
# 4. MAIN ABLATION LOOP
# ------------------------------------------------------------------
def run_config(config_key, video_path, duration_seconds):
    cfg = CONFIGS[config_key]
    
    if cfg['tracker'] == "mosse":
        det = VisionHaarMOSSE(detection_skip=cfg['base_skip'])
    else:
        det = VisionHaarCascade(detection_skip=cfg['base_skip'])
        
    scheduler = AdaptiveScheduler(cfg)
    filter_cx = TrackerKalmanFilter(gain=0.7)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(3, W)
    cap.set(4, H)
    if not cap.isOpened():
        print("[ERROR] Cannot open source:", video_path)
        return None

    out_csv = os.path.join(LOG_DIR, "per_frame_{}.csv".format(cfg['name']))
    
    # Initialization
    rows = []
    frame_idx = 0
    start_time = time.time()
    
    frames_found = 0
    tracking_lost_count = 0
    tracker_reinit_count = 0
    prev_found = False
    
    prev_cx = 0.0
    velocity = 0.0
    jitter = 0.0
    sum_jitter = 0.0
    sum_skip = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > duration_seconds:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        frame = cv2.resize(frame, (W, H))
        
        # 1. Update Scheduler (Đẩy tham số skip mới vào Detector)
        current_skip, sched_reason = scheduler.step(velocity, jitter)
        det.detection_skip = current_skip
        
        # 2. Pipeline Execution
        t_inf = time.time()
        found, bbox, cx_raw, cy_raw = det.process_frame(frame)
        inf_ms = (time.time() - t_inf) * 1000.0
        
        # 3. System Metrics Evaluation
        cx_filtered = cx_raw
        if found:
            frames_found += 1
            cx_filtered = filter_cx.update(cx_raw)
            jitter = abs(cx_raw - cx_filtered)
            sum_jitter += jitter
            velocity = abs(cx_filtered - prev_cx)
            prev_cx = cx_filtered
            
            # Logic đếm Tracker Reinit: Trúng chu kỳ Detection VÀ có tìm thấy vật thể
            if frame_idx % current_skip == 0:
                tracker_reinit_count += 1
        else:
            jitter = 0.0
            velocity = 0.0
            filter_cx.reset()

        # Logic đếm Tracking Lost
        if prev_found and not found:
            tracking_lost_count += 1
        prev_found = found

        sum_skip += current_skip
        current_fps = frame_idx / elapsed if elapsed > 0 else 0

        # Pseudo-PID error_px logic cho report
        error_px = (W/2) - cx_filtered if found else 0.0
        pid_output = error_px * 0.1 # Mock Kp
        
        # Lấy trạng thái mosse_fallback động tại runtime an toàn
        current_mosse_fallback = getattr(det, 'mosse_fallback', False)
        
        rows.append([
            frame_idx, int(found), round(cx_raw, 1), round(cx_filtered, 1), 
            round(jitter, 2), round(error_px, 1), round(pid_output, 2), 
            round(current_fps, 1), round(inf_ms, 2), current_skip, 
            round(velocity, 2), sched_reason, cfg['tracker'], int(current_mosse_fallback), cfg['name']
        ])

    cap.release()
    
    if not os.path.isdir(LOG_DIR): os.makedirs(LOG_DIR)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "found", "cx_raw", "cx_filtered", "jitter", "error_px", 
                     "pid_output", "fps", "inference_ms", "sched_skip", "sched_velocity", 
                     "sched_reason", "tracker_type", "mosse_fallback", "config_name"])
        wr.writerows(rows)

    n = frame_idx if frame_idx > 0 else 1
    fps_avg = n / elapsed if elapsed > 0 else 0
    tracking_rate = (frames_found / n) * 100.0
    jitter_mean = sum_jitter / frames_found if frames_found > 0 else 0.0
    sched_skip_mean = sum_skip / n

    summary = {
        "config_name": cfg['name'],
        "tracker_type": cfg['tracker'],
        "tracking_rate": round(tracking_rate, 2),
        "fps_avg": round(fps_avg, 2),
        "jitter_mean": round(jitter_mean, 2),
        "tracking_lost_count": tracking_lost_count,
        "tracker_reinit_count": tracker_reinit_count,
        "sched_skip_mean": round(sched_skip_mean, 2)
    }
    
    print("  -> [{}] TrRate: {:.1f}% | FPS: {:.1f} | Jitter: {:.2f} | Lost: {} | Reinit: {} | SkipMean: {:.1f}".format(
        cfg['name'], tracking_rate, fps_avg, jitter_mean, tracking_lost_count, tracker_reinit_count, sched_skip_mean))
        
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--clip", default="fast", help="Video clip to run ablation on")
    ap.add_argument("--duration", type=int, default=60, help="Run duration per config")
    args = ap.parse_args()

    v_path = os.path.join(args.videos, "{}.avi".format(args.clip))
    if not os.path.exists(v_path):
        # Fallback qua webcam nếu không có video (phục vụ test robot thực)
        v_path = 0 
        print("[WARN] Clip not found. Falling back to Webcam 0.")

    print("="*70)
    print(" ABLATION STUDY: ADAPTIVE SCHEDULER & PARETO ANALYSIS")
    print(" Target: {} | Duration: {}s".format("Webcam" if v_path == 0 else args.clip, args.duration))
    print("="*70)

    summaries = []
    for key in sorted(CONFIGS.keys()):
        s = run_config(key, v_path, args.duration)
        if s:
            summaries.append(s)

    if summaries:
        summary_path = os.path.join(LOG_DIR, "ablation_summary.csv")
        keys = ["config_name", "tracker_type", "tracking_rate", "fps_avg", 
                "jitter_mean", "tracking_lost_count", "tracker_reinit_count", "sched_skip_mean"]
        with open(summary_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summaries)
        print("\n[DONE] Summary -> {}".format(summary_path))
        print("Sẵn sàng dữ liệu cho Jupyter Notebook vẽ Pareto frontier!")

if __name__ == "__main__":
    main()