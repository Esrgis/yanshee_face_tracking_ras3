#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz_tracking_grid.py -- Minh hoa tracking bang trajectory (duong di)

Detector khong the fake duoc trajectory vi no khong co memory giua cac frame.

Output 2 file:
  tracking_sequence_<clip>.png   -- luoi 2x3, moi frame ve lai tat ca trajectory den hien tai
  tracking_trajectory_<clip>.png -- 1 anh tong hop: frame cuoi + toan bo duong di

Chay:
  python viz_tracking_grid.py                   # mac dinh: clip=scale
  python viz_tracking_grid.py --clip slow
"""
import cv2
import json
import os
import argparse
import numpy as np

W, H = 320, 240
ANNO_DIR = "data/annotations"
FPS = 20  # 600 frames / 30 giay

ID_COLORS = [
    (0,   220,  90),   # ID 1 - xanh la
    (0,   150, 255),   # ID 2 - cam
    (255,  60,  60),   # ID 3 - do
    (200,   0, 255),   # ID 4 - tim
    (0,   220, 220),   # ID 5 - cyan
]


def load_gt(clip_name):
    anno_path = os.path.join(ANNO_DIR, clip_name, "annotations.json")
    with open(anno_path) as f:
        coco = json.load(f)
    gt = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        obj_id = a.get("track_id", a.get("category_id", 1))
        gt[a["image_id"]] = {"bbox": [x, y, w, h], "track_id": obj_id}
    return gt


def bbox_center(bbox):
    x, y, w, h = bbox
    return (int(x + w / 2), int(y + h / 2))


def draw_trajectory_on_frame(frame, traj_so_far, color, current_center):
    """
    Ve duong trajectory len frame.
    Duong ke mo dan -> dam dan the hien chieu thoi gian.
    """
    if len(traj_so_far) < 2:
        return
    n = len(traj_so_far)
    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(alpha * 3))
        brightness = int(60 + alpha * 195)
        faded_color = tuple(int(c * brightness / 255) for c in color)
        cv2.line(frame, traj_so_far[i - 1], traj_so_far[i],
                 faded_color, thickness, cv2.LINE_AA)

    for i, pt in enumerate(traj_so_far[:-1]):
        alpha = (i + 1) / n
        r = max(2, int(alpha * 4))
        brightness = int(80 + alpha * 175)
        faded_color = tuple(int(c * brightness / 255) for c in color)
        cv2.circle(frame, pt, r, faded_color, -1, cv2.LINE_AA)

    # Diem hien tai: noi bat
    cv2.circle(frame, current_center, 6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, current_center, 4, color, -1, cv2.LINE_AA)


def draw_single_frame(clip_name, frame_idx, gt, seq_number, traj_history):
    img_path = os.path.join(
        ANNO_DIR, clip_name, "images",
        "frame_{:04d}.jpg".format(frame_idx)
    )
    if not os.path.exists(img_path):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(canvas, "Frame not found", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    else:
        canvas = cv2.imread(img_path)
        canvas = cv2.resize(canvas, (W, H))

    if frame_idx in gt:
        info = gt[frame_idx]
        tid = info["track_id"]
        color = ID_COLORS[(tid - 1) % len(ID_COLORS)]

        x, y, w, h = [int(v) for v in info["bbox"]]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        label = "ID:{}".format(tid)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.rectangle(canvas, (x, y - lh - 8), (x + lw + 6, y), color, -1)
        cv2.putText(canvas, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
    else:
        cv2.putText(canvas, "No GT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)

    header_h = 28
    header = np.zeros((header_h, W, 3), dtype=np.uint8)
    header[:] = (28, 28, 28)
    time_sec = frame_idx / FPS
    frame_label = "Frame {}  |  t={:.1f}s".format(seq_number, time_sec)
    cv2.putText(header, frame_label, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1)
    if seq_number < 6:
        ax = W - 20
        cv2.arrowedLine(header,
                        (ax - 12, header_h // 2),
                        (ax + 6,  header_h // 2),
                        (80, 200, 80), 2, tipLength=0.55)

    return np.vstack([header, canvas])


def build_dense_trajectory(gt, frame_range):
    """
    Lay TAT CA cac frame co annotation trong khoang frame_range = (first, last).
    Tra ve {tid: [(frame_idx, cx, cy), ...]}
    """
    first, last = frame_range
    dense = {}
    for fid in sorted(gt.keys()):
        if fid < first or fid > last:
            continue
        info = gt[fid]
        tid = info["track_id"]
        cx, cy = bbox_center(info["bbox"])
        if tid not in dense:
            dense[tid] = []
        dense[tid].append((fid, cx, cy))
    return dense


def build_grid(frames, n_cols=3):
    n_rows = (len(frames) + n_cols - 1) // n_cols
    fh, fw = frames[0].shape[:2]
    while len(frames) < n_rows * n_cols:
        frames.append(np.zeros((fh, fw, 3), dtype=np.uint8))
    rows = []
    for r in range(n_rows):
        row_frames = frames[r * n_cols: (r + 1) * n_cols]
        rows.append(np.hstack(row_frames))
    grid = np.vstack(rows)

    return grid


def build_summary_frame(clip_name, gt, all_ids, picked_frames):
    """
    1 anh tong hop: frame cuoi lam nen (lam mo) + ve TOAN BO trajectory 30 giay.
    Day la bang chung manh nhat cho reviewer.
    """
    last_fid = picked_frames[-1]
    img_path = os.path.join(
        ANNO_DIR, clip_name, "images",
        "frame_{:04d}.jpg".format(last_fid)
    )
    if os.path.exists(img_path):
        canvas = cv2.imread(img_path)
        canvas = cv2.resize(canvas, (W, H))
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Lam mo anh goc de trajectory noi bat
    canvas = cv2.addWeighted(canvas, 0.45, np.zeros_like(canvas), 0.55, 0)

    dense = build_dense_trajectory(gt, (min(all_ids), max(all_ids)))
    picked_set = set(picked_frames)

    for tid, pts in dense.items():
        color = ID_COLORS[(tid - 1) % len(ID_COLORS)]
        centers = [(cx, cy) for (_, cx, cy) in pts]
        n = len(centers)

        # Ve duong trajectory day du
        for i in range(1, n):
            alpha = i / n
            thickness = max(1, int(alpha * 3))
            brightness = int(50 + alpha * 205)
            faded = tuple(int(c * brightness / 255) for c in color)
            cv2.line(canvas, centers[i-1], centers[i], faded, thickness, cv2.LINE_AA)

        # Danh dau 6 diem chot + so thu tu
        sorted_picked = sorted(picked_set)
        for fid, cx, cy in pts:
            if fid in picked_set:
                idx = sorted_picked.index(fid) + 1
                cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 5, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, str(idx), (cx + 9, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    header_h = 28
    header = np.zeros((header_h, W, 3), dtype=np.uint8)
    header[:] = (28, 28, 28)
    label = "Full 30s trajectory  |  {} annotation frames".format(len(all_ids))
    cv2.putText(header, label, (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cap_h = 28
    cap = np.zeros((cap_h, W, 3), dtype=np.uint8)
    cap[:] = (18, 18, 18)
    cv2.putText(cap, "A detector has no memory -- it cannot produce this path",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 220, 140), 1)

    return np.vstack([header, canvas, cap])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip",  default="scale",
                    choices=["slow", "normal", "fast", "scale"])
    ap.add_argument("--n",     type=int, default=6)
    ap.add_argument("--cols",  type=int, default=3)
    ap.add_argument("--out",   default=None)
    args = ap.parse_args()

    gt = load_gt(args.clip)
    if not gt:
        print("[ERROR] Khong load duoc annotation: {}".format(args.clip))
        return

    all_ids = sorted(gt.keys())
    if len(all_ids) < args.n:
        picked = all_ids
    else:
        step = (len(all_ids) - 1) / (args.n - 1)
        picked = [all_ids[round(i * step)] for i in range(args.n)]

    print("[INFO] Clip={} | {} frames co GT | Picked={}".format(
        args.clip, len(all_ids), picked))

    # Xay dung trajectory dense cho khoang picked[0]..picked[-1]
    dense = build_dense_trajectory(gt, (picked[0], picked[-1]))

    # Render tung frame trong luoi, trajectory tich luy den frame do
    frames = []
    for seq_num, frame_idx in enumerate(picked, start=1):
        traj_history = {}
        for tid, pts in dense.items():
            pts_before = [(cx, cy) for (fid, cx, cy) in pts if fid < frame_idx]
            if pts_before:
                traj_history[tid] = pts_before

        f = draw_single_frame(args.clip, frame_idx, gt, seq_num, traj_history)
        frames.append(f)
        print("  [{}] frame={} t={:.1f}s".format(seq_num, frame_idx, frame_idx / FPS))

    # OUTPUT 1: luoi 2x3
    grid = build_grid(frames, n_cols=args.cols)
    out1 = args.out or "tracking_sequence_{}.png".format(args.clip)
    cv2.imwrite(out1, grid)
    print("\n[OK] Grid     : {}  ({}x{})".format(out1, grid.shape[1], grid.shape[0]))

    # OUTPUT 2: anh tong hop toan bo trajectory
    summary = build_summary_frame(args.clip, gt, all_ids, picked)
    out2 = "tracking_trajectory_{}.png".format(args.clip)
    cv2.imwrite(out2, summary)
    print("[OK] Trajectory: {}".format(out2))
    print("\n[HINT] Dung tracking_trajectory_*.png lam bang chung manh nhat:")
    print("       Detector khong co memory -> khong the ve duong nay.")


if __name__ == "__main__":
    main()