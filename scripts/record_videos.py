#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/record_videos.py -- Quay 3 video benchmark tren robot/laptop

Chay:
  python scripts/record_videos.py --source 0 --duration 30
  python scripts/record_videos.py --source 0 --duration 30 --output data/videos

Output:
  data/videos/slow.avi
  data/videos/normal.avi
  data/videos/fast.avi

Huong dan:
  - slow  : nguoi di cham, quay mat nhe
  - normal: di binh thuong
  - fast  : di nhanh, xoay dau nhanh
  - scale : dung im, tien sat vao camera roi lui ra xa lien tuc
"""
from __future__ import print_function
import cv2
import os
import argparse
import time

W, H  = 640, 480
FPS   = 20
CLIPS = ["slow", "normal", "fast", "scale"]


def record_one(name, source, duration, out_dir):
    out_path = os.path.join(out_dir, "{}.avi".format(name))
    cap      = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc source: {}".format(source))
        return

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    print("\n>>> BAT DAU quay: {}  ({} giay)".format(name.upper(), duration))
    print("    Di chuyen theo huong dan roi nhan SPACE de bat dau, Q de thoat.")

    # --- cho nguoi dung san sang ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (W, H))
        cv2.putText(frame, "READY - SPACE to start / Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Record", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            cap.release()
            writer.release()
            return

    # --- quay ---
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed >= duration:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (W, H))
        writer.write(frame)
        remain = int(duration - elapsed)
        cv2.putText(frame, "REC {} | {}s left".format(name.upper(), remain),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Record", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    print("    -> Saved: {}".format(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",   default="0")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--output",   default="data/videos")
    ap.add_argument("--clips",    default="slow,normal,fast")
    args = ap.parse_args()

    src   = int(args.source) if args.source.isdigit() else args.source
    clips = [c.strip() for c in args.clips.split(",")]

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    print("="*50)
    print(" RECORD {} clips x {}s each".format(len(clips), args.duration))
    print(" Output: {}".format(args.output))
    print("="*50)

    for clip in clips:
        record_one(clip, src, args.duration, args.output)

    cv2.destroyAllWindows()
    print("\nDone. Tiep theo: make benchmark")


if __name__ == "__main__":
    main()