#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_benchmark.py -- Buoc 1: So sanh 3 detector tren 4 clip (image sequence)

Chay:
  python scripts/run_benchmark.py
  python scripts/run_benchmark.py --detectors haar,lbp --clips fast,slow
  python scripts/run_benchmark.py --iou_thresh 0.4
  python scripts/run_benchmark.py --width 640 --height 480   # de chung minh LBP thua o res cao

Input:
  data/annotations/<clip>/images/frame_%04d.jpg  <- image sequence
  data/annotations/<clip>/annotations.json       <- COCO GT

Output:
  results/logs/benchmark_<detector>_<clip>.csv
  results/logs/benchmark_summary.csv
"""
from __future__ import print_function
import cv2
import csv
import time
import os
import argparse
import json
import math

# FIX #1: W, H khong hardcode nua -- lay tu args (default 320x240)
# se override trong main() sau khi parse args
W, H    = 320, 240
LOG_DIR  = "results/logs"
ANNO_DIR = "data/annotations"
CLIPS    = ["slow", "normal", "fast", "scale"]


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
    """boxA, boxB: [x, y, w, h]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / float(union)


def load_ground_truth(clip_name):
    anno_path = os.path.join(ANNO_DIR, clip_name, "annotations.json")
    if not os.path.exists(anno_path):
        print("  [INFO] Khong co GT cho clip: {}".format(clip_name))
        return None

    try:
        with open(anno_path, "r") as f:
            coco = json.load(f)
    except Exception as e:
        print("[WARNING] Loi doc GT {}: {}".format(anno_path, e))
        return None

    gt = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]] = [x, y, w, h]

    print("  [GT] {} frames co mat".format(len(gt)))
    return gt


def build_detector(name, vp, frame_w, frame_h):
    """
    FIX #2: Truyen frame_w, frame_h vao de detector biet resolution dang chay.
    Dieu nay quan trong khi upscale len 640x480 de benchmark LBP.
    min_size va max_size nen scale theo ratio so voi 320x240 goc.
    """
    scale_ratio = frame_w / 320.0  # so voi resolution goc
    min_size_scaled = int(vp.get("min_size", 30) * scale_ratio)
    max_size_scaled = int(vp.get("max_size", 400) * scale_ratio)

    COMMON = dict(
        detection_skip       = 1,
        pad_ratio            = 0.20,
        iou_reinit_threshold = 0.3,
        max_jump_px          = int(180 * scale_ratio),
        min_size             = min_size_scaled,
        max_size             = max_size_scaled,
    )

    if name == "haar":
        from core.vision_haarcascade import VisionHaarCascade
        return VisionHaarCascade(
            scale_factor  = vp.get("scale_factor_haar", 1.08),
            min_neighbors = vp.get("min_neighbors_haar", 5),
            **COMMON
        ).process_frame
    elif name == "lbp":
        from core.vision_lbp import VisionLBP
        return VisionLBP(
            scale_factor  = vp.get("scale_factor_lbp", 1.05),
            min_neighbors = vp.get("min_neighbors_lbp", 3),
            **COMMON
        ).process_frame
    elif name == "ssd":
        from core.vision_ssd import VisionSSD
        return VisionSSD(
            conf_threshold = vp.get("conf_threshold_ssd", 0.5),
            **COMMON
        ).process_frame
    else:
        raise ValueError("Unknown detector: " + name)


def run_one(detector_name, clip_name, iou_thresh, vision_params, frame_w, frame_h):
    """
    frame_w, frame_h: resolution de resize frame truoc khi dua vao detector.
    Mac dinh 320x240, co the set 640x480 de benchmark LBP thua ro hon.
    """
    img_pattern = os.path.join(ANNO_DIR, clip_name, "images", "frame_%04d.jpg")
    images_dir  = os.path.join(ANNO_DIR, clip_name, "images")

    if not os.path.isdir(images_dir):
        print("[SKIP] Khong tim thay images: {}".format(images_dir))
        return None

    out_csv = os.path.join(LOG_DIR,
        "benchmark_{}_{}_{}.csv".format(detector_name, clip_name,
                                         "{}x{}".format(frame_w, frame_h)))

    try:
        process = build_detector(detector_name, vision_params, frame_w, frame_h)
    except Exception as e:
        print("[ERROR] Build detector {}: {}".format(detector_name, e))
        return None

    # FIX #3: cv2.VideoCapture voi img_pattern doc dung image sequence
    # frame_%04d.jpg -> OpenCV tu dong doc frame_0001.jpg, frame_0002.jpg, ...
    cap = cv2.VideoCapture(img_pattern)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc sequence: {}".format(img_pattern))
        return None

    gt     = load_ground_truth(clip_name)
    has_gt = gt is not None

    rows            = []
    latency_list    = []
    frame_idx       = 0
    total_loop_time = 0.0
    total_iou       = 0.0
    max_ram         = 0.0
    tp = fp = fn = gt_frames = 0

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break  # het anh la dung, khong can duration check

        # FIX #4: resize theo frame_w, frame_h truyen vao (khong hardcode 320x240)
        frame     = cv2.resize(frame, (frame_w, frame_h))
        frame_idx += 1

        t_det        = time.time()
        found, bbox, cx, cy = process(frame)
        inference_ms = (time.time() - t_det) * 1000.0
        latency_list.append(inference_ms)

        if frame_idx == 1 or frame_idx % 10 == 0:
            ram     = get_ram_mb()
            max_ram = max(max_ram, ram)
        else:
            ram = 0.0

        frame_iou = ""
        if has_gt and frame_idx in gt:
            gt_box = gt[frame_idx]
            gt_frames += 1

            if found and bbox is not None:
                # FIX #5: scale GT bbox neu resolution khac 320x240
                scale_ratio = frame_w / 320.0
                gt_box_scaled = [
                    gt_box[0] * scale_ratio,
                    gt_box[1] * scale_ratio,
                    gt_box[2] * scale_ratio,
                    gt_box[3] * scale_ratio
                ]
                iou_val   = calculate_iou(bbox, [int(v) for v in gt_box_scaled])
                total_iou += iou_val
                frame_iou  = round(iou_val, 3)
                if iou_val >= iou_thresh:
                    tp += 1
                else:
                    fp += 1
            else:
                fn        += 1
                total_iou += 0.0
                frame_iou  = 0.0

        loop_time        = time.time() - loop_start
        total_loop_time += loop_time
        current_fps      = 1.0 / loop_time if loop_time > 0 else 0.0

        rows.append([
            frame_idx, int(found), cx, cy,
            round(inference_ms, 2), round(current_fps, 2),
            round(ram, 1) if ram > 0 else -1.0,
            frame_iou, detector_name, clip_name,
            frame_w, frame_h  # FIX: log resolution de phan biet khi so sanh
        ])

        if frame_idx % 100 == 0:
            print("  [{}@{}x{}] frame={} fps={:.1f}".format(
                detector_name, frame_w, frame_h, frame_idx, current_fps))

    cap.release()

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "found", "cx", "cy",
                     "inference_ms", "fps", "ram_mb",
                     "iou", "detector", "clip",
                     "frame_w", "frame_h"])
        wr.writerows(rows)

    n = frame_idx if frame_idx > 0 else 1

    iou_mean  = round(total_iou / gt_frames, 3) if (has_gt and gt_frames > 0) else "N/A"
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = (2*precision*recall) / (precision+recall) if (precision+recall) > 0 else 0.0
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

    summary = {
        "detector"                 : detector_name,
        "clip"                     : clip_name,
        "frame_w"                  : frame_w,   # FIX: log resolution
        "frame_h"                  : frame_h,
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
    }

    print("  -> {}@{}x{} | fps={} | P={:.3f} | R={:.3f} | F1={:.3f} | iou={} | p90={}ms".format(
        clip_name, frame_w, frame_h,
        summary["fps_avg"],
        precision, recall, f1_score,
        summary["iou_mean"], summary["inference_ms_p90"]))

    return summary


def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    vp = cfg.get("vision_params", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors",  default="haar,lbp,ssd")
    ap.add_argument("--clips",      default="slow,normal,fast,scale")
    default_iou = vp.get("iou_thresh", 0.3)
    ap.add_argument("--iou_thresh", type=float, default=default_iou)
    # FIX #6: Them width/height args de test LBP o 640x480
    ap.add_argument("--width",  type=int, default=320,
                    help="Frame width (320 or 640 to stress-test LBP)")
    ap.add_argument("--height", type=int, default=240,
                    help="Frame height (240 or 480 to stress-test LBP)")
    args = ap.parse_args()

    detectors = [d.strip() for d in args.detectors.split(",")]
    clips     = [c.strip() for c in args.clips.split(",")]

    print("=" * 60)
    print(" BENCHMARK: {} detectors x {} clips".format(
        len(detectors), len(clips)))
    print(" Resolution: {}x{}".format(args.width, args.height))
    print(" IoU threshold: {}".format(args.iou_thresh))
    print("=" * 60)

    summaries = []
    for det in detectors:
        for clip in clips:
            print("\n[{} x {} @ {}x{}]".format(
                det.upper(), clip, args.width, args.height))
            s = run_one(det, clip, args.iou_thresh, vp, args.width, args.height)
            if s:
                summaries.append(s)

    if summaries:
        summary_path = os.path.join(LOG_DIR, "benchmark_summary.csv")
        keys = ["detector", "clip", "frame_w", "frame_h",
                "frames_total", "gt_frames",
                "fps_avg", "inference_fps_theoretical",
                "inference_ms_mean", "inference_ms_std",
                "inference_ms_p50", "inference_ms_p90",
                "inference_ms_p99", "inference_ms_max",
                "precision", "recall", "f1_score",
                "iou_mean", "ram_peak", "iou_thresh"]
        with open(summary_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summaries)
        print("\nSummary -> {}".format(summary_path))

    print("\nDone. Tiep theo: make ablation")


if __name__ == "__main__":
    main()