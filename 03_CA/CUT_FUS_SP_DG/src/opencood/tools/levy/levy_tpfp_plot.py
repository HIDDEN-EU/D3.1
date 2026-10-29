"""
Plot the sparsification curves written by ``levy_tpfp_study.py``.

    python opencood/tools/levy/levy_tpfp_plot.py --dir <model_dir>/analysis_tpfp
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette (first three slots validate all-pairs) + neutrals.
SERIES = {
    "detection_score": ("#2a78d6", "detection score"),
    "logistic_B_levy": ("#eb6834", "Levy signals only (logistic, 9 signals)"),
    "logistic_C_levy_plus_score": ("#1baf7a", "Levy signals + detection score (logistic)"),
}
REFERENCE = {
    "oracle": ("#52514e", "oracle (false positives removed first)", (0, (1, 2))),
    "random": ("#8f8e88", "random removal", (0, (4, 3))),
}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(path):
    data = json.load(open(path))
    x = [100 * f for f in data["fractions"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for key, (color, label, dash) in REFERENCE.items():
        ax.plot(x, data["curves"][key], color=color, lw=1.5, ls=dash, label=label)
    for key, (color, label) in SERIES.items():
        y = data["curves"][key]
        ax.plot(x, y, color=color, lw=2, label=label)
    ax.set_xlabel("share of boxes removed, least trusted first (%)", color=MUTED)
    ax.set_ylabel("false-positive rate of the remaining boxes", color=MUTED)
    iou = os.path.basename(path).split("iou")[-1].split(".")[0]
    ax.set_title(f"Removing the least trusted predicted boxes (IoU 0.{int(iou) // 10})",
                 color=TEXT, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlim(0, 99)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    out = path.replace(".json", ".png")
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    opt = p.parse_args()
    for path in sorted(glob.glob(os.path.join(opt.dir, "sparsification_boxes_iou*.json"))):
        plot(path)


if __name__ == "__main__":
    main()
