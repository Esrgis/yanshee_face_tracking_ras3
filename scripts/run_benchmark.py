#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_benchmark.py -- Buoc 1: So sanh 3 detector tren 3 video

Chay:
  python scripts/run_benchmark.py --videos data/videos
  python scripts/run_benchmark.py --videos data/videos --detectors haar,lbp
  make benchmark

Output: results/logs/benchmark_<detector>_<clip>.csv
        results/logs/benchmark_summary.csv

Metrics do duoc:
  - fps_avg       : FPS trung binh
  - detection_ms  : thoi gian chay detector (ms/frame)
  - detect_rate   : ty le frame co face (0.0-1.0)
  - ram_mb        : RAM su dung (MB) -- chi tren Linux/Pi
"""
from __future__ import print_function
import cv2
import csv
import time
import os
import argparse
import sys

W, H    = 640, 480
LOG_DIR = "results/logs"
CLIPS = ["slow", "normal", "fast", "scale"]


# ------------------------------------------------------------------
# RAM usage (chi chay duoc tren Linux, tuc la tren Pi)
# ------------------------------------------------------------------
def get_ram_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


# ------------------------------------------------------------------
# Tao detector theo ten
# ------------------------------------------------------------------
def build_detector(name):
    """
    Tra ve ham process(frame) -> (found, bbox, cx, cy)
    De tranh phu thuoc import luc init, build inline.
    """
    if name == "haar":
        from core.vision_haarcascade import VisionHaarCascade
        det = VisionHaarCascade(detection_skip=1)   # skip=1 de do thuan detector
        return det.process_frame

    elif name == "lbp":
        from core.vision_lbp import VisionLBP
        det = VisionLBP(detection_skip=1)
        return det.process_frame

    elif name == "ssd":
        from core.vision_ssd import VisionSSD
        det = VisionSSD(detection_skip=1)
        return det.process_frame

    else:
        raise ValueError("Unknown detector: " + name)


# ------------------------------------------------------------------
# Chay 1 cap (detector x clip)
# ------------------------------------------------------------------
def run_one(detector_name, clip_name, video_path):
    if not os.path.exists(video_path):
        print("[SKIP] Khong tim thay: {}".format(video_path))
        return None

    out_csv = os.path.join(
        LOG_DIR, "benchmark_{}_{}.csv".format(detector_name, clip_name)
    )

    try:
        process = build_detector(detector_name)
    except Exception as e:
        print("[ERROR] Build detector {}: {}".format(detector_name, e))
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc: {}".format(video_path))
        return None

    rows          = []
    frame_idx     = 0
    total_det_ms  = 0.0
    total_fps     = 0.0
    detected      = 0
    prev_t        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.resize(frame, (W, H))
        frame_idx += 1

        t_det  = time.time()
        found, bbox, cx, cy = process(frame)
        det_ms = (time.time() - t_det) * 1000.0

        now    = time.time()
        dt     = now - prev_t
        fps    = 1.0 / dt if dt > 0 else 0.0
        prev_t = now

        ram    = get_ram_mb()
        if found:
            detected += 1

        total_det_ms += det_ms
        total_fps    += fps

        rows.append([
            frame_idx, int(found), cx, cy,
            round(det_ms, 2), round(fps, 2), round(ram, 1),
            detector_name, clip_name
        ])

        if frame_idx % 30 == 0:
            print("  [{}-{}] f={} fps={:.1f} found={} det_ms={:.1f}".format(
                detector_name, clip_name, frame_idx, fps, found, det_ms))

    cap.release()

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "found", "cx", "cy",
                     "detection_ms", "fps", "ram_mb",
                     "detector", "clip"])
        wr.writerows(rows)

    n = len(rows) if rows else 1
    summary = {
        "detector"    : detector_name,
        "clip"        : clip_name,
        "frames"      : frame_idx,
        "fps_avg"     : round(total_fps / n, 2),
        "detection_ms": round(total_det_ms / n, 2),
        "detect_rate" : round(detected / n, 3),
        "ram_mb"      : round(get_ram_mb(), 1),
    }

    print("  -> {} | fps={} | det_ms={} | detect_rate={}".format(
        out_csv, summary["fps_avg"],
        summary["detection_ms"], summary["detect_rate"]))

    return summary


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos",    default="data/videos")
    ap.add_argument("--detectors", default="haar,lbp,ssd")
    ap.add_argument("--clips",     default="slow,normal,fast")
    args = ap.parse_args()

    detectors = [d.strip() for d in args.detectors.split(",")]
    clips     = [c.strip() for c in args.clips.split(",")]

    print("="*50)
    print(" BENCHMARK: {} detectors x {} clips".format(
        len(detectors), len(clips)))
    print("="*50)

    summaries = []
    for det in detectors:
        for clip in clips:
            video_path = os.path.join(args.videos, "{}.avi".format(clip))
            print("\n[{} x {}]".format(det.upper(), clip))
            s = run_one(det, clip, video_path)
            if s:
                summaries.append(s)

    # --- Ghi summary ---
    if summaries:
        summary_path = os.path.join(LOG_DIR, "benchmark_summary.csv")
        keys = ["detector", "clip", "frames", "fps_avg",
                "detection_ms", "detect_rate", "ram_mb"]
        with open(summary_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summaries)
        print("\nSummary -> {}".format(summary_path))

    print("\nDone. Tiep theo: make ablation")


if __name__ == "__main__":
    main()