#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analyze_results.py - Ve figure + tinh metric cho paper (Python 3.5)

Metrics:
  avg_fps, tracking_rate, avg_jitter,
  avg_vision_ms, avg_recovery_ms  (thoi gian phuc hoi sau lost, don vi ms)

Figures:
  fig1_scatter_fps_tracking.png  Scatter fps_avg vs tracking_rate (A->E)
  fig2_bar_jitter.png            Bar chart Jitter mean
  fig3_line_cx_filtered.png      Line cx_filtered overlay (A, D, E)
  fig4_bar_sched_skip.png        Bar sched_skip_mean (C, D only)
  fig5_benchmark.png             So sanh IoU, FPS, Recall

Output: results/figures/table_summary.csv

Chay: python scripts/analyze_results.py
      make analyze
"""

from __future__ import print_function
import os, csv, argparse, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = "results/logs"
OUT_DIR = "results/figures"

STYLES = {
    "A_static_skip1"     : {"color": "#E24B4A", "ls": "-",  "label": "A: Static skip=1 (KCF)"},
    "B_static_skip5"     : {"color": "#888780", "ls": "--", "label": "B: Static skip=5 (KCF)"},
    "C_adaptive_vel_only": {"color": "#378ADD", "ls": "-.", "label": "C: Adaptive vel (KCF)"},
    "D_adaptive_full"    : {"color": "#1D9E75", "ls": "-",  "label": "D: Adaptive full (KCF)"},
    "E_static_skip5_mosse": {"color": "#9B59B6", "ls": ":",  "label": "E: Static skip=5 (MOSSE)"},
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
                # FIXED: Them sched_reason de export_table khong bi KeyError
                "sched_reason"  : row.get("sched_reason", "disabled")
            })
    return rows

def load_all_ablations(log_dir):
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

def load_benchmark_summary(log_dir):
    """Doc file benchmark_summary.csv de ve Figure 5"""
    path = os.path.join(log_dir, "benchmark_summary.csv")
    data = []
    if not os.path.exists(path):
        print("  [WARNING] Khong tim thay benchmark_summary.csv")
        return data
        
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            iou_val = row.get("iou_mean", 0.0)
            iou_float = float(iou_val) if iou_val not in ["", "N/A"] else 0.0
            
            # FIXED: Ho tro ca 'recall' va fallback xuong 'detect_rate' neu file cu
            recall_val = float(row.get("recall", row.get("detect_rate", 0.0)))
            
            data.append({
                "detector"    : row["detector"],
                "clip"        : row["clip"],
                "fps_avg"     : float(row["fps_avg"]),
                "recall"      : recall_val,
                "iou_mean"    : iou_float
            })
    print("  loaded benchmark_summary.csv")
    return data

def compute_recovery_ms(rows):
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

# ── Figures ──────────────────────────────────────────────────────────

def fig1_scatter_fps_tracking(data, out):
    """Fig 1: Scatter fps_avg vs tracking_rate cho cac config (A->E)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for key, rows in sorted(data.items()):
        s = STYLES.get(key, {})
        
        # Tinh FPS trung binh
        fps_vals = [r["fps"] for r in rows if r["fps"] > 0]
        fps_avg = np.mean(fps_vals) if fps_vals else 0.0
        
        # Tinh Tracking Rate (%)
        track_rate = (sum(r["found"] for r in rows) / len(rows)) * 100.0 if rows else 0.0
        
        # Plot point
        ax.scatter(fps_avg, track_rate, color=s.get("color","#333"), s=120, edgecolors='w', linewidth=1.5, zorder=3)
        
        # Annotate (Ky hieu A, B, C...)
        lbl = s.get("label", key).split(":")[0]
        ax.annotate(lbl, (fps_avg, track_rate), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel("Average FPS", fontsize=11)
    ax.set_ylabel("Tracking Rate (%)", fontsize=11)
    ax.set_title("Fig 1: FPS vs Tracking Rate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    fig.tight_layout()
    
    p = os.path.join(out, "fig1_scatter.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def fig2_bar_jitter(data, out):
    """Fig 2: Bar chart Jitter Mean"""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, means, colors = [], [], []
    
    for key, rows in sorted(data.items()):
        s = STYLES.get(key, {})
        j = [r["jitter"] for r in rows if r["found"] == 1]
        if not j: continue
        lbl = s.get("label", key).split(":")[0] 
        labels.append(lbl)
        means.append(np.mean(j))
        colors.append(s.get("color","#888"))
        
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.5, zorder=3)
    
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Jitter (px) - lower is better", fontsize=11)
    ax.set_title("Fig 2: Mean Tracking Jitter", fontsize=12)
    ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0)
    
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.1,
                "{:.2f}".format(m), ha="center", va="bottom", fontsize=9)
                
    fig.tight_layout()
    p = os.path.join(out, "fig2_bar_jitter.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def fig3_cx_filtered_overlay(data, out):
    """Fig 3: Line cx_filtered overlay cho config A, D, E"""
    target_keys = ["A_static_skip1", "D_adaptive_full", "E_static_skip5_mosse"]
    keys = [k for k in target_keys if k in data]
    if not keys: 
        print("  [skip fig3] Khong co data cho A, D, hoac E"); return
        
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in keys:
        s = STYLES.get(key, {})
        frames = [r["frame"] for r in data[key]]
        cxs = [r["cx_filtered"] for r in data[key]]
        
        ax.plot(frames, cxs, color=s.get("color","#333"), ls=s.get("ls","-"), 
                lw=1.5, alpha=0.85, label=s.get("label", key))
                
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("cx_filtered (px)", fontsize=11)
    ax.set_title("Fig 3: cx_filtered Over Time (A, D, E)", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    
    p = os.path.join(out, "fig3_cx_filtered.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def fig4_bar_sched_skip(data, out):
    """Fig 4: Bar chart sched_skip_mean chi cho C va D"""
    target_keys = ["C_adaptive_vel_only", "D_adaptive_full"]
    keys = [k for k in target_keys if k in data]
    if not keys: 
        print("  [skip fig4] Khong co data cho C hoac D"); return
        
    labels, means, colors = [], [], []
    for key in keys:
        s = STYLES.get(key, {})
        sk = [r["sched_skip"] for r in data[key]]
        lbl = s.get("label", key).split(":")[0]
        labels.append(lbl)
        means.append(np.mean(sk) if sk else 0.0)
        colors.append(s.get("color","#888"))
        
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.4, zorder=3)
    
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Sched Skip", fontsize=11)
    ax.set_title("Fig 4: Average sched_skip (C & D)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=0)
    
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.1,
                "{:.2f}".format(m), ha="center", va="bottom", fontsize=10)
                
    fig.tight_layout()
    p = os.path.join(out, "fig4_sched_skip.png"); fig.savefig(p, dpi=150)
    plt.close(fig); print("  saved " + p)


def fig5_benchmark(bench_data, out):
    """Figure 5: Benchmark tu benchmark_summary.csv voi Recall"""
    if not bench_data:
        return
        
    detectors = list(set(r["detector"] for r in bench_data))
    detectors.sort()
    
    avg_fps, avg_iou, avg_recall = [], [], []
    for d in detectors:
        d_rows = [r for r in bench_data if r["detector"] == d]
        avg_fps.append(np.mean([r["fps_avg"] for r in d_rows]))
        avg_iou.append(np.mean([r["iou_mean"] for r in d_rows]))
        avg_recall.append(np.mean([r["recall"] for r in d_rows])) # FIXED: dung Recall

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    x = np.arange(len(detectors))
    colors = ["#3498DB", "#E67E22", "#2ECC71"]

    # Subplot 1: FPS
    bars1 = ax1.bar(x, avg_fps, color=colors, alpha=0.8, width=0.6, zorder=3)
    ax1.set_title("Average FPS", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(detectors)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    
    # Subplot 2: Recall (FIXED)
    bars2 = ax2.bar(x, avg_recall, color=colors, alpha=0.8, width=0.6, zorder=3)
    ax2.set_title("Recall", fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(detectors)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    # Subplot 3: IoU
    bars3 = ax3.bar(x, avg_iou, color=colors, alpha=0.8, width=0.6, zorder=3)
    ax3.set_title("Mean IoU", fontsize=11)
    ax3.set_xticks(x); ax3.set_xticklabels(detectors)
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Benchmark Comparison: Performance vs Accuracy", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    p = os.path.join(out, "fig5_benchmark.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("  saved " + p)


def export_table(data, out):
    p = os.path.join(out, "table_summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "config_name", "tracker_type", "tracking_rate", "fps_avg", 
            "jitter_mean", "tracking_lost_count", 
            "scheduled_detection_successes", "sched_skip_mean"
        ])
        
        for key, rows in sorted(data.items()):
            tracker_type = "mosse" if "mosse" in key.lower() else "kcf"
            
            fps = [r["fps"] for r in rows if r["fps"] > 0]
            jit = [r["jitter"] for r in rows if r["found"] == 1]
            sk  = [r["sched_skip"] for r in rows]
            
            tracking_rate = round((sum(r["found"] for r in rows) / len(rows)) * 100, 2)
            fps_avg = round(float(np.mean(fps)), 2) if fps else 0.0
            jitter_mean = round(float(np.mean(jit)), 2) if jit else 0.0
            sched_skip_mean = round(float(np.mean(sk)), 2) if sk else 0.0
            
            rec = compute_recovery_ms(rows)
            tracking_lost_count = len(rec)
            
            sched_det_succ = 0
            for r in rows:
                if r["sched_skip"] > 0:
                    if r["sched_reason"] != "disabled" and r["found"] == 1:
                        sched_det_succ += 1
            
            w.writerow([
                key, tracker_type, tracking_rate, fps_avg, jitter_mean,
                tracking_lost_count, sched_det_succ, sched_skip_mean
            ])
            
    print("  saved " + p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=LOG_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    if not os.path.isdir(args.out_dir): os.makedirs(args.out_dir)
    
    print("Loading Ablation logs from: " + args.log_dir)
    ablation_data = load_all_ablations(args.log_dir)
    
    print("Loading Benchmark logs from: " + args.log_dir)
    bench_data = load_benchmark_summary(args.log_dir)

    print("\nGenerating figures...")
    if ablation_data:
        fig1_scatter_fps_tracking(ablation_data, args.out_dir)
        fig2_bar_jitter(ablation_data, args.out_dir)
        fig3_cx_filtered_overlay(ablation_data, args.out_dir)
        fig4_bar_sched_skip(ablation_data, args.out_dir)
        export_table(ablation_data, args.out_dir)
    else:
        print("[WARNING] Khong co data Ablation de ve Fig 1-4.")
        
    if bench_data:
        fig5_benchmark(bench_data, args.out_dir)

    print("\nDone: " + args.out_dir)

if __name__ == "__main__":
    main()