#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reorganize_repo.py
------------------
Chay mot lan tu root repo de don dep cau truc.

    python reorganize_repo.py

Script se:
  1. Tao folder structure moi
  2. Move file ve dung cho
  3. Xoa dead code (utils/, models/, exp/, experiments/)
  4. Ghi scripts/run_ablation.py   (Python 3.5 tuong thuoc)
  5. Ghi scripts/analyze_results.py (co recovery_ms metric)
  6. Tao Makefile

Sau khi chay:
    git add -A
    git commit -m "refactor: reorganize repo"
    make ablation
    make analyze
"""

from __future__ import print_function
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def p(msg):
    print(msg)


def ensure(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.isdir(path):
        os.makedirs(path)
        p("  mkdir  " + rel + "/")
    return path


def move_file(src_rel, dst_rel):
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(ROOT, dst_rel)
    if not os.path.exists(src):
        return False
    if os.path.exists(dst):
        p("  skip   " + dst_rel + " (da ton tai)")
        return False
    par = os.path.dirname(dst)
    if par and not os.path.isdir(par):
        os.makedirs(par)
    shutil.move(src, dst)
    p("  move   " + src_rel + " -> " + dst_rel)
    return True


def remove_dir_safe(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.isdir(path):
        return
    all_files = []
    for r, _, files in os.walk(path):
        for f in files:
            all_files.append(os.path.join(r, f))
    safe = all(
        f.endswith(".gitkeep") or os.path.getsize(f) == 0
        for f in all_files
    )
    if safe or not all_files:
        shutil.rmtree(path)
        p("  rmdir  " + rel + "/")
    else:
        p("  WARN   " + rel + "/ con " + str(len(all_files)) +
          " file - kiem tra thu cong")


def remove_file_safe(rel):
    path = os.path.join(ROOT, rel)
    if os.path.isfile(path):
        os.remove(path)
        p("  rm     " + rel)


def write_text(rel, content):
    path = os.path.join(ROOT, rel)
    par  = os.path.dirname(path)
    if par and not os.path.isdir(par):
        os.makedirs(par)
    with open(path, "w", encoding="utf-8") as f:  # <--- Đã sửa
        f.write(content)
    p("  write  " + rel)

# ─────────────────────────────────────────────
# Noi dung Makefile
# ─────────────────────────────────────────────

MAKEFILE = \
"""# Makefile - Yanshee Face Tracking
# Yeu cau: python trong PATH

PYTHON   = python
SRC      = 0
DURATION = 60
CONFIGS  = ABCD

run:
\t$(PYTHON) main_tracker.py

robot:
\t$(PYTHON) main_tracker_robot.py

ablation:
\t$(PYTHON) scripts/run_ablation.py --source $(SRC) --duration $(DURATION) --configs $(CONFIGS)

analyze:
\t$(PYTHON) scripts/analyze_results.py

study: ablation analyze

collect:
\t$(PYTHON) scripts/data_collector.py

.PHONY: run robot ablation analyze study collect
"""

# ─────────────────────────────────────────────
# Noi dung scripts/run_ablation.py
# ─────────────────────────────────────────────

RUN_ABLATION = \
"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
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
\"\"\"

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
                    if not self.is_tracking or self.bbox is None or \
                       self._iou(self.bbox, pad) < self.iou_thr:
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
    print("\\nDone. Tiep theo: make analyze")


if __name__ == "__main__":
    main()
"""

# ─────────────────────────────────────────────
# Noi dung scripts/analyze_results.py
# ─────────────────────────────────────────────

ANALYZE = \
"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
scripts/analyze_results.py - Ve figure + tinh metric cho paper (Python 3.5)

Metrics:
  avg_fps, tracking_rate, avg_jitter,
  avg_vision_ms, avg_recovery_ms  (thoi gian phuc hoi sau lost, don vi ms)

Figures:
  fig1_fps.png           FPS theo thoi gian
  fig2_jitter.png        Jitter box plot
  fig3_adaptive_skip.png Skip dong (C & D)
  fig4_recovery.png      Recovery time bar chart

Output: results/figures/table_summary.csv  <- dan vao paper

Chay: python scripts/analyze_results.py
      make analyze
\"\"\"

from __future__ import print_function
import os, csv, argparse, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "results/logs"
OUT_DIR = "results/figures"

STYLES = {
    "A_static_skip1"     : {"color": "#E24B4A", "ls": "-",  "label": "A: Static skip=1"},
    "B_static_skip5"     : {"color": "#888780", "ls": "--", "label": "B: Static skip=5"},
    "C_adaptive_vel_only": {"color": "#378ADD", "ls": "-.", "label": "C: Adaptive vel"},
    "D_adaptive_full"    : {"color": "#1D9E75", "ls": "-",  "label": "D: Adaptive full"},
}


def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame"         : int(row["frame"]),
                "found"         : int(row["found"]),
                "cx_raw"        : float(row["cx_raw"]),
                "cx_filtered"   : float(row["cx_filtered"]),
                "jitter"        : float(row["jitter"]),
                "fps"           : float(row["fps"]),
                "vision_ms"     : float(row["vision_ms"]),
                "sched_skip"    : int(row["sched_skip"]),
                "sched_velocity": float(row["sched_velocity"]),
                "config_name"   : row["config_name"],
            })
    return rows


def load_all(log_dir):
    data = {}
    if not os.path.isdir(log_dir):
        print("[ERROR] Khong tim thay: " + log_dir); return data
    for fname in sorted(os.listdir(log_dir)):
        if not fname.startswith("ablation_") or not fname.endswith(".csv"): continue
        rows = load_csv(os.path.join(log_dir, fname))
        if rows:
            key = rows[0]["config_name"]
            data[key] = rows
            print("  loaded {} ({} frames)".format(fname, len(rows)))
    return data


def compute_recovery_ms(rows):
    \"\"\"
    Tinh thoi gian phuc hoi (ms) sau moi episode lost.

    Lost episode: found chuyen 1->0 roi 0->1.
    Moi frame lost dong gop: 1000/fps ms.
    Tra ve list cac recovery_ms (moi phan tu = 1 episode).
    \"\"\"
    out = []; lost = False; acc = 0.0
    for r in rows:
        fms = (1000.0 / r["fps"]) if r["fps"] > 0 else 0.0
        if not lost and r["found"] == 0:
            lost = True;  acc  = fms
        elif lost and r["found"] == 0:
            acc += fms
        elif lost and r["found"] == 1:
            out.append(round(acc, 2)); lost = False; acc = 0.0
    return out


def smooth(vals, w=20):
    out = []; buf = collections.deque(maxlen=w)
    for v in vals:
        buf.append(v); out.append(sum(buf)/len(buf))
    return out


def stat(lst, fn):
    return round(fn(lst), 2) if lst else "N/A"


# ── Figures ──────────────────────────────────────────────────────────

def fig1_fps(data, out):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for key, rows in data.items():
        s = STYLES.get(key, {})
        ax.plot([r["frame"] for r in rows],
                smooth([r["fps"] for r in rows]),
                color=s.get("color","#333"), ls=s.get("ls","-"),
                lw=1.5, label=s.get("label", key))
    ax.set_xlabel("Frame", fontsize=11); ax.set_ylabel("FPS", fontsize=11)
    ax.set_title("Frame rate - ablation configs", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3, lw=0.5); ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = os.path.join(out, "fig1_fps.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def fig2_jitter(data, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, vals, colors = [], [], []
    for key, rows in data.items():
        s = STYLES.get(key, {})
        j = [r["jitter"] for r in rows if r["found"] == 1]
        if not j: continue
        lbl = s.get("label", key)
        for pf in ("A: ","B: ","C: ","D: "): lbl = lbl.replace(pf,"")
        labels.append(lbl); vals.append(j); colors.append(s.get("color","#888"))
    bp = ax.boxplot(vals, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Jitter (px) - lower is better", fontsize=11)
    ax.set_title("Tracking jitter (tracked frames only)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, lw=0.5); fig.tight_layout()
    p = os.path.join(out, "fig2_jitter.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def fig3_skip(data, out):
    keys = [k for k in ["C_adaptive_vel_only","D_adaptive_full"] if k in data]
    if not keys: print("  [skip fig3] khong co adaptive data"); return
    fig, ax = plt.subplots(figsize=(8, 3))
    for key in keys:
        s = STYLES.get(key, {})
        ax.plot([r["frame"] for r in data[key]],
                [r["sched_skip"] for r in data[key]],
                color=s.get("color","#333"), ls=s.get("ls","-"),
                lw=1.2, alpha=0.8, label=s.get("label", key))
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("detection_skip", fontsize=11)
    ax.set_title("Adaptive detection_skip over time", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_yticks(list(range(1,16))); fig.tight_layout()
    p = os.path.join(out, "fig3_adaptive_skip.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def fig4_recovery(data, out):
    \"\"\"Bar chart: avg recovery time (ms) - metric chinh cua adaptive scheduler.\"\"\"
    labels, means, stds, colors = [], [], [], []
    for key, rows in data.items():
        s   = STYLES.get(key, {})
        rec = compute_recovery_ms(rows)
        if not rec: rec = [0.0]
        labels.append(s.get("label", key).split(":")[0])
        means.append(float(np.mean(rec)))
        stds.append(float(np.std(rec)))
        colors.append(s.get("color","#888"))
    fig, ax = plt.subplots(figsize=(6, 4))
    x    = list(range(len(labels)))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8,
                  width=0.5, error_kw={"elinewidth":1.2, "ecolor":"#444"})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Avg recovery time (ms) - lower is better", fontsize=11)
    ax.set_title("Target recovery time after lost event", fontsize=12)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 5,
                "{:.0f}ms".format(m), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "fig4_recovery.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def export_table(data, out):
    p = os.path.join(out, "table_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config","Label",
                    "Avg FPS","Median FPS","Tracking Rate (%)",
                    "Avg Jitter (px)","Median Jitter (px)",
                    "Avg Vision (ms)","Avg Skip",
                    "Avg Recovery (ms)","Median Recovery (ms)","N Lost Events"])
        for key, rows in data.items():
            s   = STYLES.get(key, {})
            fps = [r["fps"]        for r in rows if r["fps"] > 0]
            jit = [r["jitter"]     for r in rows if r["found"] == 1]
            vis = [r["vision_ms"]  for r in rows]
            sk  = [r["sched_skip"] for r in rows]
            rec = compute_recovery_ms(rows)
            pct = round(sum(r["found"] for r in rows) / len(rows) * 100, 1)
            w.writerow([key, s.get("label", key),
                        stat(fps, np.mean), stat(fps, np.median), pct,
                        stat(jit, np.mean), stat(jit, np.median),
                        stat(vis, np.mean), stat(sk, np.mean),
                        stat(rec, np.mean), stat(rec, np.median), len(rec)])
    print("  saved " + p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=LOG_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    if not os.path.isdir(args.out_dir): os.makedirs(args.out_dir)
    print("Loading from: " + args.log_dir)
    data = load_all(args.log_dir)
    if not data:
        print("[ERROR] Khong co CSV. Chay: make ablation"); return
    print("\\nGenerating figures...")
    fig1_fps(data, args.out_dir)
    fig2_jitter(data, args.out_dir)
    fig3_skip(data, args.out_dir)
    fig4_recovery(data, args.out_dir)
    export_table(data, args.out_dir)
    print("\\nDone: " + args.out_dir)


if __name__ == "__main__":
    main()
"""


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    p("=" * 55)
    p(" Repo reorganizer - Yanshee Face Tracking")
    p("=" * 55)

    p("\n[1] Tao folders...")
    for d in ["core", "hardware", "scripts",
              "results/logs", "results/figures", "data/videos"]:
        ensure(d)

    p("\n[2] Move experiment files -> scripts/...")
    for src, dst in [
        ("exp/run_ablation.py",           "scripts/run_ablation.py"),
        ("exp/analyze_results.py",        "scripts/analyze_results.py"),
        ("run_ablation.py",               "scripts/run_ablation.py"),
        ("analyze_results.py",            "scripts/analyze_results.py"),
        ("experiments/run_ablation.py",   "scripts/run_ablation.py"),
        ("experiments/analyze_results.py","scripts/analyze_results.py"),
    ]:
        move_file(src, dst)

    p("\n[3] Move CSV logs -> results/logs/...")
    for old in ["data/logs", "data/log"]:
        old_path = os.path.join(ROOT, old)
        if os.path.isdir(old_path):
            for f in os.listdir(old_path):
                if f.endswith(".csv"):
                    move_file(os.path.join(old, f),
                              os.path.join("results", "logs", f))

    p("\n[4] Xoa dead code...")
    remove_dir_safe("utils")
    remove_dir_safe("models")
    remove_dir_safe("exp")
    remove_dir_safe("experiments")
    remove_dir_safe("data/logs")
    remove_file_safe("test_haar.py")

    p("\n[5] Ghi scripts/run_ablation.py...")
    write_text("scripts/run_ablation.py", RUN_ABLATION)

    p("\n[6] Ghi scripts/analyze_results.py...")
    write_text("scripts/analyze_results.py", ANALYZE)

    p("\n[7] Ghi scripts/__init__.py...")
    write_text("scripts/__init__.py", "")

    p("\n[8] Ghi Makefile...")
    write_text("Makefile", MAKEFILE)

    p("\n" + "=" * 55)
    p(" Xong! Lenh tiep theo:")
    p("=" * 55)
    p("  git add -A")
    p("  git commit -m \"refactor: reorganize repo\"")
    p("")
    p("  make ablation           # chay 4 config x 60s")
    p("  make analyze            # ve figure + tinh metric")
    p("  make study              # ablation + analyze lien tiep")
    p("")
    p("  make ablation SRC=data/videos/test.avi DURATION=90")
    p("  make ablation CONFIGS=AB    # chi chay A va B")


if __name__ == "__main__":
    main()
