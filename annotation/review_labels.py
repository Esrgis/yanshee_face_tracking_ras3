# annotation/review_labels.py
# Chay tren laptop Python 3.10
# Xem tung frame, sua/xoa box sai, luu thanh GT chinh thuc

import cv2
import json
import os
import argparse

VALID_CLIPS = ["slow", "normal", "fast", "scale"]

HELP = """
PHIM TAT:
  A / <- : frame truoc
  D / -> : frame sau
  X      : xoa box frame nay (danh dau khong co mat)
  R      : khoi phuc box goc tu YOLO (neu da xoa)
  S      : luu va thoat
  Q      : thoat KHONG luu
  E      : chinh sua box thu cong (keo chuot)
"""


def load_raw(anno_dir, clip_name):
    raw_path = os.path.join(anno_dir, "{}_raw.json".format(clip_name))
    if not os.path.exists(raw_path):
        print("[ERROR] Khong tim thay: {}".format(raw_path))
        return None
    with open(raw_path, "r") as f:
        data = json.load(f)
    return data.get("frames", {})


def save_gt(annotations, anno_dir, clip_name):
    out_path = os.path.join(anno_dir, "{}.json".format(clip_name))
    with open(out_path, "w") as f:
        json.dump({"frames": annotations}, f, indent=2)
    print("[SAVED] {}".format(out_path))


def draw_box(frame, bbox, color=(0, 255, 0), label=""):
    if bbox is None:
        return frame
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


class BoxDrawer:
    """Cho phep ve box moi bang chuot."""
    def __init__(self):
        self.drawing  = False
        self.start    = None
        self.end      = None
        self.new_box  = None

    def reset(self):
        self.drawing = False
        self.start   = None
        self.end     = None
        self.new_box = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start   = (x, y)
            self.end     = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end     = (x, y)
            x1 = min(self.start[0], self.end[0])
            y1 = min(self.start[1], self.end[1])
            w  = abs(self.end[0] - self.start[0])
            h  = abs(self.end[1] - self.start[1])
            if w > 5 and h > 5:
                self.new_box = [x1, y1, w, h]

    def get_preview(self, frame):
        if self.drawing and self.start and self.end:
            disp = frame.copy()
            cv2.rectangle(disp, self.start, self.end, (0, 165, 255), 2)
            return disp
        return frame


def review(clip_name, video_path, anno_dir):
    raw_anno = load_raw(anno_dir, clip_name)
    if raw_anno is None:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc: {}".format(video_path))
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(HELP)
    print("[INFO] {} | {} frames total".format(clip_name, total_frames))

    # Doc tat ca frame vao bo nho (video ngan ~600 frames, 320x240 = ok)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Working copy de sua — giu raw de restore
    working = {}
    for k, v in raw_anno.items():
        working[k] = v

    # Chi hien thi frame co trong annotation
    anno_keys    = sorted(raw_anno.keys(), key=lambda x: int(x))
    current_idx  = 0  # index trong anno_keys
    edit_mode    = False
    drawer       = BoxDrawer()

    win_name = "Review: {}".format(clip_name)
    cv2.namedWindow(win_name)

    while True:
        if not anno_keys:
            print("[WARN] Khong co frame nao duoc annotate.")
            break

        key_str    = anno_keys[current_idx]
        frame_num  = int(key_str)
        bbox       = working.get(key_str)

        # Lay frame tuong ung
        if frame_num - 1 < len(frames):
            disp = frames[frame_num - 1].copy()
        else:
            disp = None

        if disp is None:
            current_idx = min(current_idx + 1, len(anno_keys) - 1)
            continue

        # Scale len x2 de de nhin
        disp = cv2.resize(disp, (640, 480))
        scale_x = 640.0 / frames[0].shape[1]
        scale_y = 480.0 / frames[0].shape[0]

        # Ve box hien tai
        if bbox is not None:
            scaled = [
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y),
            ]
            draw_box(disp, scaled, (0, 255, 0), "YOLO")
        else:
            cv2.putText(disp, "NO FACE (deleted)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Ve box dang ve (edit mode)
        if edit_mode:
            disp = drawer.get_preview(disp)
            if drawer.new_box is not None:
                # Convert toa do scaled ve goc
                nb = drawer.new_box
                working[key_str] = [
                    int(nb[0] / scale_x),
                    int(nb[1] / scale_y),
                    int(nb[2] / scale_x),
                    int(nb[3] / scale_y),
                ]
                drawer.reset()
                edit_mode = False

        # HUD
        status = "OK" if bbox is not None else "DELETED"
        color  = (0, 255, 0) if bbox is not None else (0, 0, 255)
        cv2.putText(disp,
            "Frame {}/{} | idx {}/{} | {}".format(
                frame_num, len(frames),
                current_idx + 1, len(anno_keys),
                status),
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(disp,
            "A/D=nav | X=del | R=restore | E=edit | S=save | Q=quit",
            (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if edit_mode:
            cv2.putText(disp, "EDIT MODE - Keo chuot de ve box moi",
                        (5, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            cv2.setMouseCallback(win_name, drawer.mouse_callback)
        else:
            cv2.setMouseCallback(win_name, lambda *a: None)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(30) & 0xFF

        if key in [ord('d'), 83]:    # D hoac ->
            current_idx = min(current_idx + 1, len(anno_keys) - 1)
        elif key in [ord('a'), 81]:  # A hoac <-
            current_idx = max(current_idx - 1, 0)
        elif key == ord('x') or key == ord('X'):
            working[key_str] = None
            print("[DEL] frame {}".format(frame_num))
        elif key == ord('r') or key == ord('R'):
            working[key_str] = raw_anno.get(key_str)
            print("[RESTORE] frame {}".format(frame_num))
        elif key == ord('e') or key == ord('E'):
            edit_mode = True
            drawer.reset()
            print("[EDIT] Keo chuot de ve box moi cho frame {}".format(frame_num))
        elif key == ord('s') or key == ord('S'):
            save_gt(working, anno_dir, clip_name)
            break
        elif key == ord('q') or key == ord('Q'):
            print("[QUIT] Khong luu.")
            break

    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips",      default="slow,normal,fast,scale")
    ap.add_argument("--videos",     default="data/videos")
    ap.add_argument("--anno_dir",   default="data/annotations")
    args = ap.parse_args()

    clips = [c.strip() for c in args.clips.split(",")]

    for clip in clips:
        if clip not in VALID_CLIPS:
            print("[SKIP] {}".format(clip))
            continue
        video_path = os.path.join(args.videos, "{}.avi".format(clip))
        if not os.path.exists(video_path):
            print("[SKIP] Khong co video: {}".format(video_path))
            continue
        raw_path = os.path.join(args.anno_dir, "{}_raw.json".format(clip))
        if not os.path.exists(raw_path):
            print("[SKIP] Chua co raw annotation: {}".format(raw_path))
            continue
        print("\n=== REVIEW: {} ===".format(clip.upper()))
        review(clip, video_path, args.anno_dir)

    print("\n[ALL DONE] GT files -> data/annotations/")


if __name__ == "__main__":
    main()