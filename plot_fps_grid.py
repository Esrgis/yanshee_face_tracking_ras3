#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_fps_grid.py -- Ve luoi 2x2 FPS theo frame cho 4 cau hinh x 4 clip

Chay:
  python plot_fps_grid.py
  python plot_fps_grid.py --data_dir results/logs --out fps_grid.png
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CONFIGS = [
    ("B_static_kcf_skip5",   "B: KCF Static",    "#185FA5", "-"),
    ("D_adaptive_kcf_skip5", "D: KCF Adaptive",  "#993C1D", "--"),
    ("E_static_mosse_skip5", "E: MOSSE Static",  "#3B6D11", "-"),
    ("F_adaptive_mosse_skip5","F: MOSSE Adaptive","#854F0B", "--"),
]
CLIPS = ["slow", "normal", "fast", "scale"]
CLIP_LABELS = {
    "slow"  : "Slow",
    "normal": "Normal",
    "fast"  : "Fast",
    "scale" : "Scale",
}
WINDOW = 15  # rolling average de lam muot duong FPS


def load_fps(data_dir, config, clip):
    fname = "per_frame_{}_{}.csv".format(config, clip)
    path  = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        print("[WARN] Khong tim thay: {}".format(path))
        return None
    df = pd.read_csv(path)
    return df["fps"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="results/logs")
    ap.add_argument("--out",      default="fps_grid.png")
    ap.add_argument("--dpi",      type=int, default=200)
    ap.add_argument("--window",   type=int, default=WINDOW)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)


    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, clip in zip(axes_flat, CLIPS):
        for config, label, color, ls in CONFIGS:
            fps = load_fps(args.data_dir, config, clip)
            if fps is None:
                continue

            frames = np.arange(1, len(fps) + 1)

            # Raw FPS mo nhat
            ax.plot(frames, fps,
                    color=color, alpha=0.15, linewidth=0.8, linestyle=ls)

            # Rolling average ro hon
            series  = pd.Series(fps)
            fps_avg = series.rolling(args.window, min_periods=1, center=True).mean()
            ax.plot(frames, fps_avg,
                    color=color, alpha=0.9, linewidth=1.6,
                    linestyle=ls, label=label)

        ax.set_title(CLIP_LABELS[clip], fontsize=11, fontweight="500")
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_ylabel("FPS", fontsize=9)
        ax.set_xlim(1, 600)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.1,  linewidth=0.4)
        ax.tick_params(labelsize=8)

    # Legend chung o duoi
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.93)  # thêm dòng này

    fig.legend(handles, labels,
            loc="lower center", ncol=4,
            fontsize=9, framealpha=0.9,
            bbox_to_anchor=(0.5, 0.01))  # đổi -0.04 -> 0.01

    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("Saved -> {}".format(args.out))
    plt.show()


if __name__ == "__main__":
    main()