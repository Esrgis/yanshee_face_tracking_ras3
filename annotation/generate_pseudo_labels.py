# annotation/generate_pseudo_labels.py
# Chay tren laptop Python 3.10, KHONG len Pi
# pip install ultralytics opencv-python

import cv2
import json
import os
import argparse
from ultralytics import YOLO

VALID_CLIPS = ["slow", "normal", "fast", "scale"]


def generate(video_path, clip_name, out_dir, conf_thresh=0.5, sample_every=1):
    """
    Chay YOLOv8-face tren video, xuat pseudo-label JSON.

    sample_every: chi lay 1 frame moi N frame
                  - video 30s x 20fps = 600 frames
                  - sample_every=1 -> anno tat ca 600 frames
                  - sample_every=2 -> anno 300 frames (du cho benchmark)
    """
    if not os.path.exists(video_path):
        print("[ERROR] Khong tim thay: {}".format(video_path))
        return

    model = YOLO("yolov8n-face.pt")  # auto download lan dau

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc video: {}".format(video_path))
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] {} | {} frames | sample_every={}".format(
        clip_name, total_frames, sample_every))

    annotations = {}
    frame_idx   = 0
    anno_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % sample_every != 0:
            continue

        results = model(frame, conf=conf_thresh, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf[0])
                boxes.append({
                    "bbox": [x1, y1, w, h],
                    "conf": round(conf, 3)
                })

        # Chi luu frame co mat (bo qua background frame)
        if boxes:
            # Neu co nhieu mat -> lay cai conf cao nhat (single-object assumption)
            best = max(boxes, key=lambda b: b["conf"])
            annotations[str(frame_idx)] = best["bbox"]
            anno_count += 1

        if frame_idx % 100 == 0:
            print("  [{}/{}] annotated={}".format(
                frame_idx, total_frames, anno_count))

    cap.release()

    # Xuat raw JSON
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    raw_path = os.path.join(out_dir, "{}_raw.json".format(clip_name))
    with open(raw_path, "w") as f:
        json.dump({"frames": annotations}, f, indent=2)

    print("[DONE] {} frames annotated -> {}".format(anno_count, raw_path))
    print("[NEXT] Chay review_labels.py de kiem tra va sua tay.")
    return raw_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos",       default="data/videos",      help="Thu muc chua video")
    ap.add_argument("--out",          default="data/annotations", help="Thu muc xuat GT")
    ap.add_argument("--clips",        default="slow,normal,fast,scale")
    ap.add_argument("--conf",         type=float, default=0.5,    help="Confidence threshold")
    ap.add_argument("--sample_every", type=int,   default=1,      help="Lay 1 frame moi N frame")
    args = ap.parse_args()

    clips = [c.strip() for c in args.clips.split(",")]

    print("=" * 55)
    print(" GENERATE PSEUDO LABELS - YOLOv8-face")
    print(" clips={} | conf={} | sample_every={}".format(
        clips, args.conf, args.sample_every))
    print("=" * 55)

    for clip in clips:
        if clip not in VALID_CLIPS:
            print("[SKIP] clip khong hop le: {}".format(clip))
            continue
        video_path = os.path.join(args.videos, "{}.avi".format(clip))
        generate(video_path, clip, args.out,
                 conf_thresh=args.conf,
                 sample_every=args.sample_every)

    print("\n[ALL DONE] Raw annotations -> {}".format(args.out))
    print("Buoc tiep theo: python annotation/review_labels.py")


if __name__ == "__main__":
    main()