#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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
    """
    Tinh thoi gian phuc hoi (ms) sau moi episode lost.

    Lost episode: found chuyen 1->0 roi 0->1.
    Moi frame lost dong gop: 1000/fps ms.
    Tra ve list cac recovery_ms (moi phan tu = 1 episode).
    """
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
    """Bar chart: avg recovery time (ms) - metric chinh cua adaptive scheduler."""
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
    print("\nGenerating figures...")
    fig1_fps(data, args.out_dir)
    fig2_jitter(data, args.out_dir)
    fig3_skip(data, args.out_dir)
    fig4_recovery(data, args.out_dir)
    export_table(data, args.out_dir)
    print("\nDone: " + args.out_dir)


if __name__ == "__main__":
    main()
