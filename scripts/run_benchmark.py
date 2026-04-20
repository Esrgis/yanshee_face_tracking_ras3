#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_benchmark.py -- Buoc 1: So sanh 3 detector tren 3 video

Chay:
  python scripts/run_benchmark.py --videos data/videos
  python scripts/run_benchmark.py --videos data/videos --iou_thresh 0.4

Metrics (THEO SPEC LOCKED FINAL):
  - fps_avg                   : FPS trung binh toan video (System FPS)
  - inference_fps_theoretical : Tốc độ suy luận lý thuyết của rieng detector
  - inference_ms_mean         : Thoi gian inference trung binh (ms)
  - inference_ms_std, p50...  : Phan phoi do tre (Latency distribution)
  - frames_total, gt_frames   : Tong so frame va so frame thuc te co mat nguoi
  - precision, recall, f1     : Danh gia chat luong detection
  - iou_mean                  : Trung binh IoU (Chi tinh tren frames co GT)
  - ram_peak                  : RAM su dung cao nhat (MB) - Sampled every 10 frames
"""
from __future__ import print_function
import cv2
import csv
import time
import os
import argparse
import json
import math 

W, H    = 640, 480
LOG_DIR = "results/logs"
CLIPS = ["slow", "normal", "fast", "scale"]

def get_ram_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def load_ground_truth(clip_name):
    gt_path = os.path.join("data/annotations", "{}.json".format(clip_name))
    if not os.path.exists(gt_path):
        return None
    try:
        with open(gt_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print("[WARNING] Loi doc file GT {}: {}".format(gt_path, e))
        return None

def build_detector(name):
    # [IMPLEMENTER NOTE]: Khong cham vao core/
    if name == "haar":
        from core.vision_haarcascade import VisionHaarCascade
        det = VisionHaarCascade(detection_skip=1)
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

def run_one(detector_name, clip_name, video_path, iou_thresh):
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
    cap.set(3, W) 
    cap.set(4, H)
    
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc: {}".format(video_path))
        return None

    gt_data = load_ground_truth(clip_name)
    has_gt = gt_data is not None
    if has_gt:
        print("  [INFO] Da load Ground Truth (IoU Threshold: {})".format(iou_thresh))

    rows            = []
    latency_list    = [] 
    frame_idx       = 0
    total_loop_time = 0.0
    total_iou       = 0.0
    max_ram         = 0.0
    
    tp = fp = fn = 0
    gt_frames    = 0 # [IMPLEMENTER FIX]: Rename thanh gt_frames

    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (W, H))
        frame_idx += 1

        t_det = time.time()
        found, bbox, cx, cy = process(frame)
        inference_ms = (time.time() - t_det) * 1000.0 # [IMPLEMENTER FIX]: Rename tu pipeline_ms
        latency_list.append(inference_ms)

        # [IMPLEMENTER FIX]: Toi uu RAM sampling, tranh overhead tren Raspberry Pi 3
        if frame_idx == 1 or frame_idx % 10 == 0:
            ram = get_ram_mb()
            max_ram = max(max_ram, ram)
        else:
            ram = 0.0 # Khong do de tiet kiem I/O, nhung trong log file per-frame se the hien la 0.0

        frame_iou = ""
        if has_gt:
            gt_box = gt_data.get(str(frame_idx))
            
            if gt_box: 
                gt_frames += 1
                
                if found and bbox is not None:
                    iou_val = calculate_iou(bbox, gt_box)
                    total_iou += iou_val
                    frame_iou = round(iou_val, 3)
                    
                    if iou_val >= iou_thresh: 
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1 
                    total_iou += 0.0 
                    frame_iou = 0.0 
            else:
                if found:
                    fp += 1 
                    frame_iou = 0.0

        loop_time = time.time() - loop_start
        total_loop_time += loop_time
        current_fps = 1.0 / loop_time if loop_time > 0 else 0.0

        rows.append([
            frame_idx, int(found), cx, cy,
            round(inference_ms, 2), round(current_fps, 2), round(ram, 1) if ram > 0 else -1.0,
            frame_iou, detector_name, clip_name
        ])

    cap.release()

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        # [IMPLEMENTER FIX]: Cap nhat ten header
        wr.writerow(["frame", "found", "cx", "cy",
                     "inference_ms", "fps", "ram_mb", "iou",
                     "detector", "clip"])
        wr.writerows(rows)

    n = frame_idx if frame_idx > 0 else 1
    
    if has_gt and gt_frames > 0:
        iou_mean = round(total_iou / gt_frames, 3)
    else:
        iou_mean = "N/A"

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    fps_avg   = n / total_loop_time if total_loop_time > 0 else 0.0

    latency_list.sort()
    n_lat = len(latency_list)
    
    if n_lat > 0:
        inf_mean = sum(latency_list) / n_lat
        inf_max  = latency_list[-1]
        inf_p50  = latency_list[min(int(n_lat * 0.50), n_lat - 1)]
        inf_p90  = latency_list[min(int(n_lat * 0.90), n_lat - 1)]
        inf_p99  = latency_list[min(int(n_lat * 0.99), n_lat - 1)]
        
        variance = sum((x - inf_mean) ** 2 for x in latency_list) / n_lat
        inf_std  = math.sqrt(variance)
        
        inference_fps_theo = 1000.0 / inf_mean if inf_mean > 0 else 0.0
    else:
        inf_mean = inf_max = inf_p50 = inf_p90 = inf_p99 = inf_std = inference_fps_theo = 0.0

    # [IMPLEMENTER FIX]: Loai bo raw_detect_rate theo Spec
    summary = {
        "detector"                 : detector_name,
        "clip"                     : clip_name,
        "frames_total"             : n,
        "gt_frames"                : gt_frames if has_gt else "N/A",
        "fps_avg"                  : round(fps_avg, 2),
        "inference_fps_theoretical": round(inference_fps_theo, 2),
        "inference_ms_mean"        : round(inf_mean, 2),
        "inference_ms_std"         : round(inf_std, 2),
        "inference_ms_p50"         : round(inf_p50, 2),
        "inference_ms_p90"         : round(inf_p90, 2),
        "inference_ms_p99"         : round(inf_p99, 2),
        "inference_ms_max"         : round(inf_max, 2),
        "precision"                : round(precision, 3) if has_gt else "N/A",
        "recall"                   : round(recall, 3) if has_gt else "N/A",
        "f1_score"                 : round(f1_score, 3) if has_gt else "N/A",
        "iou_mean"                 : iou_mean,
        "ram_peak"                 : round(max_ram, 1),
        "iou_thresh"               : iou_thresh,
        "mode"                     : "detection_only"
    }

    print("  -> {} | theo_fps={} | P={:.3f} | R={:.3f} | F1={:.3f} | iou={} | p90_ms={}".format(
        out_csv, summary["inference_fps_theoretical"], 
        precision, recall, f1_score, summary["iou_mean"], summary["inference_ms_p90"]))

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos",    default="data/videos")
    ap.add_argument("--detectors", default="haar,lbp,ssd")
    ap.add_argument("--clips",     default="slow,normal,fast,scale")
    ap.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for TP/FP")
    args = ap.parse_args()

    detectors = [d.strip() for d in args.detectors.split(",")]
    clips     = [c.strip() for c in args.clips.split(",")]

    print("="*60)
    print(" BENCHMARK: {} detectors x {} clips".format(len(detectors), len(clips)))
    print(" IOU THRESHOLD: {}".format(args.iou_thresh))
    print("="*60)

    summaries = []
    for det in detectors:
        for clip in clips:
            video_path = os.path.join(args.videos, "{}.avi".format(clip))
            print("\n[{} x {}]".format(det.upper(), clip))
            s = run_one(det, clip, video_path, args.iou_thresh)
            if s:
                summaries.append(s)

    if summaries:
        summary_path = os.path.join(LOG_DIR, "benchmark_summary.csv")
        # [IMPLEMENTER FIX]: Cap nhat fieldnames final
        keys = ["detector", "clip", "frames_total", "gt_frames", "fps_avg", 
                "inference_fps_theoretical", "inference_ms_mean", "inference_ms_std", 
                "inference_ms_p50", "inference_ms_p90", "inference_ms_p99", "inference_ms_max", 
                "precision", "recall", "f1_score", 
                "iou_mean", "ram_peak", "iou_thresh", "mode"]
        with open(summary_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summaries)
        print("\nSummary -> {}".format(summary_path))

    print("\nDone. Tiep theo: make ablation")

if __name__ == "__main__":
    main()