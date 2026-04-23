#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz_bbox.py -- Xem anh + GT bbox sau khi scale len 640x480

Chay:
  python viz_bbox.py                        # mac dinh: clip=scale, frame=1
  python viz_bbox.py --clip slow --frame 50
  python viz_bbox.py --clip scale --browse  # bam phim de xem tung frame
"""
import cv2
import json
import os
import argparse

W, H     = 640, 480
ANNO_DIR = "data/annotations"


def load_gt_scaled(clip_name):
    anno_path = os.path.join(ANNO_DIR, clip_name, "annotations.json")
    with open(anno_path) as f:
        coco = json.load(f)
    gt = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]] = [x * 2.0, y * 2.0, w * 2.0, h * 2.0]
    return gt


def draw_frame(clip_name, frame_idx, gt):
    img_path = os.path.join(ANNO_DIR, clip_name, "images",
                            "frame_{:04d}.jpg".format(frame_idx))
    if not os.path.exists(img_path):
        print("[ERROR] Khong tim thay: {}".format(img_path))
        return None

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (W, H))

    # Ve GT bbox (xanh la)
    if frame_idx in gt:
        x, y, w, h = [int(v) for v in gt[frame_idx]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "GT", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "No GT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Label thong tin
    label = "clip={} | frame={} | {}x{}".format(clip_name, frame_idx, W, H)
    cv2.putText(frame, label, (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip",   default="scale",
                    choices=["slow", "normal", "fast", "scale"])
    ap.add_argument("--frame",  type=int, default=1)
    ap.add_argument("--browse", action="store_true",
                    help="Duyet tung frame: [d] next, [a] prev, [q] quit")
    args = ap.parse_args()

    gt = load_gt_scaled(args.clip)
    print("[GT] {} frames co annotation".format(len(gt)))
    print("[INFO] Phim: [d] next | [a] prev | [q] quit")

    if args.browse:
        idx = args.frame
        max_idx = max(gt.keys()) if gt else 600
        while True:
            frame = draw_frame(args.clip, idx, gt)
            if frame is None:
                break
            cv2.imshow("viz_bbox", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                idx = min(idx + 1, max_idx)
            elif key == ord('a'):
                idx = max(idx - 1, 1)
    else:
        frame = draw_frame(args.clip, args.frame, gt)
        if frame is not None:
            cv2.imshow("viz_bbox", frame)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()