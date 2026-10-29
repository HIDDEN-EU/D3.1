"""
Average precision when the Levy mask's proposals are added to the detections.

Proposals only fill gaps -- cells the mask flags that no predicted box covers --
and are ranked below every real detection, so they extend the tail of the
precision-recall curve rather than reordering it.

    python opencood/tools/levy/levy_proposal_ap.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy \\
        --prefix levy_intermediate_prop_204.8_102.4_epoch1_use_cav4
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from opencood.tools.levy.levy_collect import match_boxes
from opencood.tools.levy.levy_proposals import ProposalRecorder
from opencood.utils.eval_utils import calculate_ap

IOUS = (0.3, 0.5, 0.7)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--min_cells", nargs="*", type=int, default=[6, 10, 15, 20])
    p.add_argument("--min_prob", nargs="*", type=float, default=[0.0])
    return p.parse_args()


def average_precision(frames, geometry, proposals, corners_of, min_cells, min_prob,
                      floor=None):
    """AP per IoU threshold, with the kept proposals appended to each frame."""
    stat = {t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in IOUS}
    n_added = 0
    for frame, rows in frames.groupby("frame"):
        gt = geometry[f"gt_{frame}"]
        pred = geometry[f"pred_{frame}"]
        scores = rows["score"].to_numpy()
        boxes = [pred]
        # Proposals sit below the weakest detection ANYWHERE in the run, ordered
        # among themselves by the mask probability. Average precision ranks boxes
        # across frames, so a per-frame floor would let one frame's proposal
        # outrank another frame's real detection.
        floor = floor if floor is not None else (scores.min() if len(scores) else 0.1)
        extra_scores = np.zeros(0)
        if proposals is not None:
            take = proposals[(proposals["frame"] == frame)
                             & (proposals["n_cells"] >= min_cells)
                             & (proposals["prob"] >= min_prob)]
            if len(take):
                boxes.append(corners_of(take))
                extra_scores = floor * 0.99 * (
                    0.5 + 0.5 * take["prob"].to_numpy() / max(take["prob"].max(), 1e-9))
                n_added += len(take)
        all_boxes = np.concatenate(boxes) if len(boxes) > 1 else pred
        all_scores = np.concatenate([scores, extra_scores])
        _, tp, _ = match_boxes(
            torch.tensor(all_boxes) if len(all_boxes) else None,
            torch.tensor(all_scores) if len(all_scores) else None,
            torch.tensor(gt) if len(gt) else None,
            IOUS,
        )
        for t in IOUS:
            stat[t]["gt"] += len(gt)
            stat[t]["score"].extend(all_scores.tolist())
            stat[t]["tp"].extend(tp[t].tolist())
            stat[t]["fp"].extend((1 - tp[t]).tolist())
    return {t: calculate_ap(stat, t)[0] for t in IOUS}, n_added


def main():
    opt = parse_args()
    base = os.path.join(opt.model_dir, opt.prefix)
    frames = pd.read_csv(f"{base}_boxes.csv")
    geometry = np.load(f"{base}_geometry.npz")
    raw = np.load(f"{base}_proposals.npz")
    proposals = pd.DataFrame({k: raw[k] for k in raw.files
                              if raw[k].shape[:1] == raw["frame"].shape})
    rec = ProposalRecorder([0, 0, float(raw["box_z"][0]) * 2, 0, 0, 0], None,
                           dict(zip("hwl", raw["box_size_hwl"])))

    def corners_of(rows):
        return rec.corners({k: rows[k].to_numpy() for k in ("x", "y", "yaw")})

    floor = float(frames["score"].min())
    print(f"proposals ranked below every detection (lowest detection score {floor:.4f})")
    plain, _ = average_precision(frames, geometry, None, corners_of, 0, 0.0)
    print(f"detections only:        " + "  ".join(f"AP{int(t * 100)} {plain[t]:.4f}" for t in IOUS))
    print(f"\n{'min cells':>9} {'min prob':>8} {'proposals':>10}   " +
          "  ".join(f"{'AP' + str(int(t * 100)):>16}" for t in IOUS))
    for min_cells in opt.min_cells:
        for min_prob in opt.min_prob:
            ap, n = average_precision(frames, geometry, proposals, corners_of,
                                      min_cells, min_prob, floor)
            print(f"{min_cells:>9} {min_prob:>8.2f} {n:>10}   " + "  ".join(
                f"{ap[t]:>8.4f} ({ap[t] - plain[t]:+.4f})" for t in IOUS))


if __name__ == "__main__":
    main()
