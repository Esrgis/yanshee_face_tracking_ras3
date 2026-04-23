#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_ablation.py -- Buoc 2: Danh gia Scheduler & Tracking Trade-off (Pareto Frontier)

Chay:
  python scripts/run_ablation.py
  python scripts/run_ablation.py --clips fast,scale --configs A,B,D
  python scripts/run_ablation.py --clips fast --max_frames 300

Input:
  data/annotations/<clip>/images/frame_%04d.jpg  <- image sequence (giong benchmark)
  config.json                                     <- doc configs tu day, khong hardcode

Output:
  results/logs/per_frame_<config_name>_<clip>.csv
  results/logs/ablation_summary.csv
"""
from __future__ import print_function
import cv2
import csv
import time
import os
import argparse
import json

# FIX #1: KHONG import TrackerKalmanFilter tu core.filters o day nua
# vi class thuc su tu core se bi overwrite boi local definition
# Thay vao do: import dung class thuc su voi numpy/cv2 KalmanFilter
from core.filters import TrackerKalmanFilter   # class that voi cv2.KalmanFilter
from core.adaptive_scheduler import AdaptiveDetectionScheduler
from core.state_machine import RobotState      # can de mock TRACKING state

W, H    = 320, 240
LOG_DIR  = "results/logs"
ANNO_DIR = "data/annotations"
CLIPS    = ["slow", "normal", "fast", "scale"]

# ------------------------------------------------------------------
# 1. CORE INJECTION (STRICT GUARDRAIL: DO NOT MODIFY CORE FOLDER)
# ------------------------------------------------------------------
from core.vision_haarcascade import VisionHaarCascade

class VisionHaarMOSSE(VisionHaarCascade):
    """
    Subclass tiem MOSSE Tracker truc tiep tai Runtime.
    Khong lam thay doi ma nguon goc cua he thong.
    """
    def __init__(self, detection_skip=1, **kwargs):
        super(VisionHaarMOSSE, self).__init__(detection_skip=detection_skip, **kwargs)
        self.mosse_fallback = False

    def _init_tracker(self, frame, bbox):
        # FIX: mirror logic class cha -- set self.bbox sau khi init thanh cong
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False
        try:
            try:
                self.tracker = cv2.TrackerMOSSE_create()
                self.mosse_fallback = False
            except AttributeError:
                self.tracker = cv2.legacy.TrackerMOSSE_create()
                self.mosse_fallback = True

            ok = self.tracker.init(frame, (x, y, w, h))
            if ok:
                self.is_tracking = True
                self.bbox        = (x, y, w, h)  # FIX: class cha co dong nay, MOSSE phai co
                return True
            else:
                self._reset_tracker()
                return False
        except Exception as e:
            print("[MOSSE] Init error:", e)
            self._reset_tracker()
            return False


# ------------------------------------------------------------------
# 2. ADAPTIVE SCHEDULER WRAPPER
# ------------------------------------------------------------------
# FIX: Dung AdaptiveDetectionScheduler tu core thay vi re-implement local.
# Khi chay offline (khong co robot), mock robot_state = TRACKING de scheduler
# khong bi override cung ve skip=1 (logic FSM override chi dung tren robot that).
# Config "static" (A, B, E) dung enabled=False -> luon tra base_skip.

class AblationSchedulerWrapper:
    """
    Thin wrapper quanh AdaptiveDetectionScheduler tu core.
    Cho phep chay ablation offline khong can FSM that.
    """
    def __init__(self, cfg):
        self.is_adaptive = cfg['adaptive']
        self.base_skip   = cfg['base_skip']
        # Mock robot state: luon TRACKING de tranh FSM override khi chay offline
        self._mock_state = RobotState.TRACKING

        self._sched = AdaptiveDetectionScheduler(
            enabled         = cfg['adaptive'],
            base_skip       = cfg['base_skip'],
            min_skip        = 1,
            max_skip        = cfg['base_skip'],   # max = base de A/B test rõ rang
            alpha           = cfg['alpha'],
            beta            = cfg['beta'],
            velocity_window = 5,
        )

    def step(self, cx_filtered, cy_filtered, jitter):
        """
        Goi compute_skip voi mock RobotState.TRACKING.
        cx_filtered, cy_filtered: toa do Kalman output (-1 neu khong co target).
        Returns: (skip, reason_str)
        """
        skip   = self._sched.compute_skip(cx_filtered, cy_filtered, jitter, self._mock_state)
        reason = self._sched.last_reason
        return skip, reason

    def reset(self):
        self._sched.reset()


# ------------------------------------------------------------------
# 3. ABLATION CONFIGS -- doc tu config.json, fallback ve hardcode
# ------------------------------------------------------------------
def load_ablation_configs(config_json_path):
    """
    FIX #2: Doc configs tu config.json thay vi hardcode.
    Dam bao beta D = 0.2 (khop voi production), khong phai 0.05.
    """
    try:
        with open(config_json_path) as f:
            cfg = json.load(f)
        sched_cfg = cfg.get("adaptive_scheduler", {}).get("configs", {})
    except Exception as e:
        print("[WARN] Khong doc duoc config.json ({}), dung fallback hardcode.".format(e))
        sched_cfg = {}

    # Map tu config.json sang CONFIGS dict day du
    # config.json chi co base_skip/alpha/beta, can them: name, adaptive, tracker
    TRACKER_MAP = {
        "A": ("kcf",   False),
        "B": ("kcf",   False),
        "C": ("kcf",   True),
        "D": ("kcf",   True),
        "E": ("mosse", False),
        "F": ("mosse", True),   # MOSSE + adaptive (same params as D) -- de so sanh fair
    }

    CONFIGS = {}
    for key, (tracker, adaptive) in TRACKER_MAP.items():
        if key in sched_cfg:
            c = sched_cfg[key]
            base_skip = c.get("base_skip", 5)
            alpha     = c.get("alpha", 0.0)
            beta      = c.get("beta",  0.0)
        else:
            # Fallback khi config.json khong co key nay (vi du key E, F)
            fallback = {
                "A": (1,   0.0,  0.0),
                "B": (5,   0.0,  0.0),
                "C": (5,   0.08, 0.0),
                "D": (5,   0.08, 0.2),
                "E": (5,   0.0,  0.0),
                "F": (5,   0.08, 0.2),  # MOSSE + adaptive full, same hyperparams as D
            }
            base_skip, alpha, beta = fallback[key]

        suffix = "mosse" if tracker == "mosse" else "kcf"
        adaptive_str = "adaptive" if adaptive else "static"
        CONFIGS[key] = {
            "name"     : "{}_{}_{}_skip{}".format(key, adaptive_str, suffix, base_skip),
            "adaptive" : adaptive,
            "base_skip": base_skip,
            "alpha"    : alpha,
            "beta"     : beta,
            "tracker"  : tracker,
        }

    return CONFIGS


# ------------------------------------------------------------------
# 4. MAIN ABLATION LOOP
# ------------------------------------------------------------------
def run_config(config_key, clip_name, CONFIGS, kalman_params, max_frames=None):
    cfg = CONFIGS[config_key]

    # Build detector
    det_kwargs = dict(detection_skip=cfg['base_skip'])
    if cfg['tracker'] == "mosse":
        det = VisionHaarMOSSE(**det_kwargs)
    else:
        det = VisionHaarCascade(**det_kwargs)

    # FIX: Dung AblationSchedulerWrapper bao quan AdaptiveDetectionScheduler tu core
    scheduler = AblationSchedulerWrapper(cfg)

    kf_cx = TrackerKalmanFilter(
        process_noise    = kalman_params.get("process_noise_cov", 0.03),
        measurement_noise= kalman_params.get("measurement_noise_cov", 0.1)
    )
    kalman_initialized = False

    # FIX #4: Doc image sequence giong run_benchmark.py
    img_pattern = os.path.join(ANNO_DIR, clip_name, "images", "frame_%04d.jpg")
    images_dir  = os.path.join(ANNO_DIR, clip_name, "images")

    if not os.path.isdir(images_dir):
        print("[ERROR] Khong tim thay images dir: {}".format(images_dir))
        return None

    cap = cv2.VideoCapture(img_pattern)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc sequence: {}".format(img_pattern))
        return None

    out_csv = os.path.join(LOG_DIR,
        "per_frame_{}_{}.csv".format(cfg['name'], clip_name))

    # -- Init metrics --
    rows               = []
    frame_idx          = 0
    frames_found       = 0
    tracking_lost_count= 0
    tracker_reinit_count = 0
    prev_found         = False

    prev_cx_filtered   = float(W) / 2.0  # init o giua frame
    velocity           = 0.0
    jitter             = 0.0
    sum_jitter         = 0.0
    sum_skip           = 0
    total_inf_ms       = 0.0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # FIX #5: het anh thi dung, KHONG dung elapsed > duration

        frame_idx += 1

        # FIX #6: max_frames thay the --duration cho image sequence
        if max_frames is not None and frame_idx > max_frames:
            break

        frame = cv2.resize(frame, (W, H))

        # 1. Update Scheduler
        # FIX: AblationSchedulerWrapper.step() nhan (cx_filtered, cy_filtered, jitter)
        # Lan dau chua co cx_filtered -> dung -1 de scheduler biet chua co target
        cx_for_sched = cx_filtered if kalman_initialized else -1
        cy_for_sched = cy_filtered if kalman_initialized else -1
        current_skip, sched_reason = scheduler.step(cx_for_sched, cy_for_sched, jitter)
        det.detection_skip = current_skip

        # 2. Detection/Tracking pipeline
        t_inf              = time.time()
        found, bbox, cx_raw, cy_raw = det.process_frame(frame)
        inf_ms             = (time.time() - t_inf) * 1000.0
        total_inf_ms      += inf_ms

        # 3. Kalman filter update
        # FIX #7: Dung dung API cua TrackerKalmanFilter that:
        #   init_state() lan dau, sau do predict() + update()
        if found:
            frames_found += 1

            if not kalman_initialized:
                kf_cx.init_state(cx_raw, cy_raw)
                kalman_initialized = True
                cx_filtered = cx_raw
                cy_filtered = cy_raw
            else:
                # predict truoc, roi correct bang measurement
                kf_cx.predict()
                cx_filtered, cy_filtered = kf_cx.update(cx_raw, cy_raw)

            jitter    = abs(cx_raw - cx_filtered)
            velocity  = scheduler._sched.last_velocity  # lay tu scheduler, khong tinh tay
            sum_jitter += jitter
            prev_cx_filtered = float(cx_filtered)

            # FIX #8: Tracker reinit dem chinh xac hon:
            # chi dem khi DUNG vao chu ky detection (modulo) VA co mat vat the
            # Day la ap xim hop ly nhat ma khong sua core
            if current_skip > 0 and frame_idx % current_skip == 0:
                tracker_reinit_count += 1

        else:
            # mat target: reset kalman va scheduler history
            kalman_initialized = False
            scheduler.reset()
            jitter   = 0.0
            velocity = 0.0
            cx_filtered = W // 2
            cy_filtered = H // 2

        # Dem tracking lost (transition found->not found)
        if prev_found and not found:
            tracking_lost_count += 1
        prev_found = found

        sum_skip += current_skip
        elapsed   = time.time() - start_time
        current_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        # Pseudo-PID (mock, chi de log)
        error_px   = (W / 2.0) - cx_filtered if found else 0.0
        pid_output = error_px * 0.1

        mosse_fallback = int(getattr(det, 'mosse_fallback', False))

        rows.append([
            frame_idx, int(found),
            round(cx_raw, 1) if found else -1,
            round(cx_filtered, 1),
            round(jitter, 2), round(error_px, 1), round(pid_output, 2),
            round(current_fps, 1), round(inf_ms, 2),
            current_skip, round(velocity, 2),
            sched_reason, cfg['tracker'], mosse_fallback,
            cfg['name'], clip_name
        ])

    cap.release()
    elapsed = time.time() - start_time

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "frame", "found", "cx_raw", "cx_filtered",
            "jitter", "error_px", "pid_output",
            "fps", "inference_ms", "sched_skip", "sched_velocity",
            "sched_reason", "tracker_type", "mosse_fallback",
            "config_name", "clip"
        ])
        wr.writerows(rows)

    n            = frame_idx if frame_idx > 0 else 1
    fps_avg      = n / elapsed if elapsed > 0 else 0.0
    tracking_rate= (frames_found / n) * 100.0
    jitter_mean  = sum_jitter / frames_found if frames_found > 0 else 0.0
    sched_skip_mean = sum_skip / n

    summary = {
        "config_name"        : cfg['name'],
        "clip"               : clip_name,
        "tracker_type"       : cfg['tracker'],
        "tracking_rate"      : round(tracking_rate, 2),
        "fps_avg"            : round(fps_avg, 2),
        "jitter_mean"        : round(jitter_mean, 2),
        "tracking_lost_count": tracking_lost_count,
        "tracker_reinit_count": tracker_reinit_count,
        "sched_skip_mean"    : round(sched_skip_mean, 2),
        "frames_total"       : n,
    }

    print("  -> [{}][{}] TrRate:{:.1f}% | FPS:{:.1f} | Jitter:{:.2f} | Lost:{} | Reinit:{} | SkipMean:{:.1f}".format(
        cfg['name'], clip_name,
        tracking_rate, fps_avg, jitter_mean,
        tracking_lost_count, tracker_reinit_count, sched_skip_mean))

    return summary


def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.json")

    CONFIGS = load_ablation_configs(config_path)

    # Doc kalman params tu config.json
    try:
        with open(config_path) as f:
            full_cfg = json.load(f)
        kalman_params = full_cfg.get("filter_kalman", {})
    except Exception:
        kalman_params = {}

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips",      default="slow,normal,fast,scale",
                    help="Clips to run (comma-separated)")
    ap.add_argument("--configs",    default=",".join(sorted(CONFIGS.keys())),
                    help="Config keys to run, e.g. A,B,D")
    # FIX #9: max_frames thay the duration -- phu hop voi image sequence
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Max frames per clip per config (None = all frames)")
    args = ap.parse_args()

    clips       = [c.strip() for c in args.clips.split(",")]
    config_keys = [k.strip() for k in args.configs.split(",")]

    print("=" * 70)
    print(" ABLATION STUDY: ADAPTIVE SCHEDULER & PARETO ANALYSIS")
    print(" Clips: {} | Configs: {}".format(clips, config_keys))
    if args.max_frames:
        print(" Max frames per run: {}".format(args.max_frames))
    print("=" * 70)

    summaries = []
    for clip in clips:
        print("\n--- Clip: {} ---".format(clip))
        for key in sorted(config_keys):
            if key not in CONFIGS:
                print("[SKIP] Config key '{}' khong ton tai.".format(key))
                continue
            s = run_config(key, clip, CONFIGS, kalman_params, args.max_frames)
            if s:
                summaries.append(s)

    if summaries:
        summary_path = os.path.join(LOG_DIR, "ablation_summary.csv")
        keys = ["config_name", "clip", "tracker_type",
                "tracking_rate", "fps_avg",
                "jitter_mean", "tracking_lost_count",
                "tracker_reinit_count", "sched_skip_mean",
                "frames_total"]
        if not os.path.isdir(LOG_DIR):
            os.makedirs(LOG_DIR)
        with open(summary_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summaries)
        print("\n[DONE] Summary -> {}".format(summary_path))
        print("Ready for Jupyter Notebook to plot Pareto frontier!")


if __name__ == "__main__":
    main()