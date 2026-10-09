"""
Better boxes for the Levy mask's proposals, and whether they lift AP50 / AP70.

A proposal is a group of flagged cells that no detection covers. The recorded
box (``blob``: centroid, principal axis, anchor size) finds the car but places it
loosely. This tool builds the other boxes offline from a run recorded with
``--save_proposals`` (which also keeps each group's cells and the detector's
regressed box at its strongest anchor):

* ``weighted``  probability-weighted centroid and axis, anchor size;
* ``rect``      a car-sized rectangle placed to cover the most mask probability;
* ``split``     groups larger than one car split by weighted k-means, each fitted
                as ``rect`` (one proposal per part);
* ``detector``  the detector's own regressed box at the group's strongest anchor
                (its class probability was below the score threshold);
* ``hybrid``    ``detector`` when its box covers at least ``HYBRID_SHARE`` of the mask
                probability ``rect`` covers, else ``rect``.

For each variant it reports how the boxes sit on the ground-truth cars the
detections missed (IoU, centre / heading / size error, merged groups), and AP
with the proposals ranked below every detection. The order among proposals is
either the mask probability or a small logistic fitted on the ``--fit_prefix``
run's proposals (label: IoU >= 0.5 with a missed car), never on the evaluated run.

    python opencood/tools/levy/levy_proposal_refine.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy \\
        --fit_prefix levy_intermediate_prop2_204.8_102.4_val_epoch1_use_cav4 \\
        --prefix levy_intermediate_prop2_204.8_102.4_epoch1_use_cav4
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from opencood.tools.levy.levy_collect import match_boxes
from opencood.tools.levy.levy_proposals import group_elongation
from opencood.utils import box_utils, common_utils
from opencood.utils.eval_utils import calculate_ap

IOUS = (0.3, 0.5, 0.7)
VARIANTS = ("blob", "weighted", "rect", "split", "detector", "hybrid")
SPLIT_FACTOR = 1.5  # split a group holding more than this many cars' worth of cells
MAX_GROUP_CELLS = 1100  # ~700 m^2 (~80 car footprints on the 0.8 m grid): a region, not a car
MAX_PARTS = 8  # at most this many cars from one group when splitting
FIT_CELLS = 2000  # a group larger than this is fitted on a probability-weighted sample of its cells
CHUNK = 256  # candidate rectangles evaluated at once (bounds memory for large groups)
HYBRID_SHARE = 0.8  # use the detector box if it covers >= this share of the mask mass the rect covers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", default=None, help="run evaluated")
    p.add_argument("--fit_prefix", required=True,
                   help="run whose proposals fit the ranking model and the car size")
    p.add_argument("--min_cells", type=int, default=6)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--write_selector", default=None,
                   help="instead of evaluating: fit the detector-box quality model on the "
                        "--fit_prefix run and write it, with the --keep threshold, for "
                        "inference_heter_in_order.py --levy_proposals")
    p.add_argument("--keep", type=float, default=0.2,
                   help="share of the fit run's proposals the written threshold keeps")
    return p.parse_args()


def write_selector(opt):
    """Fit the quality model for detector-box proposals and save it for live use."""
    import json

    boxes, geo, props, meta = load_run(opt.model_dir, opt.fit_prefix, opt.min_cells)
    var = build_variant(props, "detector", meta, detected_sizes(boxes, geo))
    y = (iou_to_gt(var, geo, missed_gt(boxes, geo))["iou"] >= 0.5).to_numpy(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(feature_matrix(var), y)
    quality = model.predict_proba(feature_matrix(var))[:, 1]
    scaler, logistic = model
    params = {
        "fitted_on": opt.fit_prefix,
        "target": "detector-box proposal overlaps a missed car at IoU >= 0.5",
        "n_fit": int(len(y)), "positive_rate": float(y.mean()),
        "features": FEATURES,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": logistic.coef_[0].tolist(), "intercept": float(logistic.intercept_[0]),
        "keep_share": opt.keep,
        "threshold": float(np.quantile(quality, 1 - opt.keep)),
        "min_cells": opt.min_cells, "max_group_cells": MAX_GROUP_CELLS,
        "score_scale": 0.05,
        "note": "kept proposals are scored score_scale * quality, below every re-scored detection",
    }
    with open(opt.write_selector, "w") as f:
        json.dump(params, f, indent=1)
    print(f"wrote {opt.write_selector}: {len(y)} fit proposals ({y.mean():.1%} positive), "
          f"threshold {params['threshold']:.4f} keeps {opt.keep:.0%}")


# ----------------------------------------------------------------------------- data


def load_run(model_dir, prefix, min_cells):
    base = os.path.join(model_dir, prefix)
    boxes = pd.read_csv(f"{base}_boxes.csv")
    geometry = np.load(f"{base}_geometry.npz")
    raw = np.load(f"{base}_proposals.npz")
    props = pd.DataFrame({k: raw[k] for k in raw.files if raw[k].shape[:1] == raw["frame"].shape})
    cells = np.load(f"{base}_proposal_cells.npz")
    off = cells["offset"]
    grid, pc = cells["grid"], cells["pc_range"]
    H, W = int(grid[0]), int(grid[1])
    dx, dy = (pc[3] - pc[0]) / W, (pc[4] - pc[1]) / H
    keep = ((props["n_cells"] >= min_cells) & (props["n_cells"] <= MAX_GROUP_CELLS)).to_numpy()
    print(f"{prefix}: {len(props)} proposals, {int((props['n_cells'] > MAX_GROUP_CELLS).sum())} "
          f"groups over {MAX_GROUP_CELLS} cells left out (regions, not cars)")
    rows, cols, prob = cells["rows"], cells["cols"], cells["prob"]
    sel = np.nonzero(keep)[0]
    props = props[keep].reset_index(drop=True)
    props["cells"] = [np.stack([pc[0] + (cols[off[i]:off[i + 1]] + 0.5) * dx,
                                pc[1] + (rows[off[i]:off[i + 1]] + 0.5) * dy], 1) for i in sel]
    props["cell_prob"] = [prob[off[i]:off[i + 1]].astype(float) for i in sel]
    meta = {"hwl": raw["box_size_hwl"], "z": float(raw["box_z"][0]), "cell_area": dx * dy}
    return boxes, geometry, props, meta


def detected_sizes(boxes, geometry):
    """Median (h, w, l) of the run's own detections: the car size a rectangle uses."""
    sizes = []
    for frame in boxes["frame"].unique():
        pred = geometry[f"pred_{frame}"]
        if len(pred):
            sizes.append(box_utils.corner_to_center(pred, order="hwl")[:, 3:6])
    return np.median(np.concatenate(sizes), 0)


# ------------------------------------------------------------------------ box fits


def principal(xy, w):
    c = np.average(xy, axis=0, weights=w)
    if len(xy) < 3:
        return c, 0.0
    cov = np.cov((xy - c).T, aweights=w)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, int(np.argmax(vals))]
    return c, float(np.arctan2(major[1], major[0]))


def _inside(xy, cand, half_l, half_w):
    """(M, N) bool: cell centres ``xy`` (N, 2) inside rectangles ``cand`` (M, 3: x, y, yaw)."""
    d = xy[None] - cand[:, None, :2]
    c, s = np.cos(cand[:, 2])[:, None], np.sin(cand[:, 2])[:, None]
    lon = c * d[..., 0] + s * d[..., 1]
    lat = -s * d[..., 0] + c * d[..., 1]
    return (np.abs(lon) <= half_l) & (np.abs(lat) <= half_w)


def fit_rect(xy, w, length, width):
    """Rectangle (x, y, yaw) covering the most mask probability; ties go to the one
    nearest the weighted centroid. Coarse grid, then a local refinement."""
    if len(xy) > FIT_CELLS:
        pick = np.random.default_rng(0).choice(len(xy), FIT_CELLS, replace=False, p=w / w.sum())
        xy, w = xy[pick], np.ones(FIT_CELLS)
    c0, _ = principal(xy, w)
    half_l, half_w = length / 2, width / 2

    def search(centre, yaws, span, step):
        off = np.arange(-span, span + 1e-6, step)
        ox, oy, oyaw = np.meshgrid(off, off, yaws, indexing="ij")
        cand = np.stack([centre[0] + ox.ravel(), centre[1] + oy.ravel(), oyaw.ravel()], 1)
        mass = np.concatenate([_inside(xy, cand[i:i + CHUNK], half_l, half_w) @ w
                               for i in range(0, len(cand), CHUNK)])
        dist = np.hypot(cand[:, 0] - c0[0], cand[:, 1] - c0[1])
        best = np.argmax(mass - 1e-3 * dist * w.sum())
        return cand[best]

    coarse = search(c0, np.deg2rad(np.arange(0, 180, 10)), 1.5, 0.3)
    fine = search(coarse[:2], coarse[2] + np.deg2rad(np.arange(-7.5, 7.6, 2.5)), 0.3, 0.1)
    return fine, float((_inside(xy, fine[None], half_l, half_w)[0] * w).sum() / w.sum())


def build_variant(props, name, meta, car_hwl):
    """One row per proposal box of ``name``: frame, box (x, y, z, h, w, l, yaw), features."""
    h, wd, ln = car_hwl
    car_cells = wd * ln / meta["cell_area"]
    out = []
    for i, p in props.iterrows():
        xy, w = p["cells"], p["cell_prob"]
        base = {"src": i, "frame": p["frame"], "n_cells": p["n_cells"], "prob": p["prob"],
                "prob_max": p["prob_max"], "det_prob": p.get("det_prob", np.nan)}
        ah, aw, al = meta["hwl"]
        if name == "blob":
            boxes = [((p["x"], p["y"], meta["z"], ah, aw, al, p["yaw"]), 1.0)]
        elif name == "weighted":
            c, yaw = principal(xy, w)
            boxes = [((c[0], c[1], meta["z"], ah, aw, al, yaw), 1.0)]
        elif name in ("rect", "split", "hybrid"):
            parts = [(xy, w)]
            if name == "split" and p["n_cells"] > SPLIT_FACTOR * car_cells:
                k = min(int(round(p["n_cells"] / car_cells)), MAX_PARTS)
                lab = KMeans(k, n_init=3, random_state=0).fit(xy, sample_weight=w).labels_
                parts = [(xy[lab == j], w[lab == j]) for j in range(k) if (lab == j).sum() >= 3]
            boxes = []
            for pxy, pw in parts:
                r, frac = fit_rect(pxy, pw, ln, wd)
                boxes.append(((r[0], r[1], meta["z"], h, wd, ln, r[2]), frac))
            if name == "hybrid" and "det_x" in p:
                det = np.array([[p["det_x"], p["det_y"], p["det_yaw"]]])
                det_frac = float((_inside(xy, det, p["det_l"] / 2, p["det_w"] / 2)[0] * w).sum() / w.sum())
                if det_frac >= HYBRID_SHARE * boxes[0][1]:
                    boxes = [((p["det_x"], p["det_y"], p["det_z"], p["det_h"], p["det_w"],
                               p["det_l"], p["det_yaw"]), det_frac)]
        elif name == "detector":
            boxes = [((p["det_x"], p["det_y"], p["det_z"], p["det_h"], p["det_w"],
                       p["det_l"], p["det_yaw"]), 1.0)]
        elong = group_elongation(xy, w)
        for box, frac in boxes:
            out.append({**base, "box": np.array(box, dtype=np.float32), "fit_frac": frac,
                        "elong": elong,
                        "dist": float(np.hypot(box[0], box[1])), "n_parts": len(boxes)})
    return pd.DataFrame(out)


def corners(boxes7):
    if not len(boxes7):
        return np.zeros((0, 8, 3), np.float32)
    return box_utils.boxes_to_corners_3d(torch.tensor(np.stack(boxes7), dtype=torch.float32),
                                         order="hwl").numpy()


# ------------------------------------------------------------------ labels & report


def missed_gt(boxes, geometry):
    """frame -> bool array over GT: not matched by any detection at IoU 0.3."""
    out = {}
    for frame, rows in boxes.groupby("frame"):
        gt, pred = geometry[f"gt_{frame}"], geometry[f"pred_{frame}"]
        _, _, hit = match_boxes(torch.tensor(pred) if len(pred) else None,
                                torch.tensor(rows["score"].to_numpy()) if len(pred) else None,
                                torch.tensor(gt) if len(gt) else None, (0.3,))
        out[frame] = hit[0.3] == 0
    return out


def iou_to_gt(var, geometry, missed):
    """Per proposal box: best IoU with a missed GT car, that car's centre/heading/size error,
    and how many missed cars the box touches."""
    rows = []
    for frame, g in var.groupby("frame"):
        gt = geometry[f"gt_{frame}"]
        m = missed.get(frame, np.zeros(len(gt), bool))
        gt_m = gt[m] if len(gt) else gt
        pc = corners(list(g["box"]))
        if not len(gt_m):
            rows += [dict(idx=i, iou=0.0, touch=0) for i in g.index]
            continue
        gp = list(common_utils.convert_format(gt_m))
        gc = box_utils.corner_to_center(gt_m, order="hwl")
        for k, (i, poly) in enumerate(zip(g.index, common_utils.convert_format(pc))):
            ious = common_utils.compute_iou(poly, gp)
            j = int(np.argmax(ious))
            b = g.at[i, "box"]
            d = b[:2] - gc[j, :2]
            c, s = np.cos(gc[j, 6]), np.sin(gc[j, 6])
            rows.append(dict(idx=i, iou=float(ious[j]), touch=int((ious > 0).sum()),
                             lon=c * d[0] + s * d[1], lat=-s * d[0] + c * d[1],
                             dyaw=float(np.rad2deg(abs((b[6] - gc[j, 6] + np.pi / 2) % np.pi - np.pi / 2))),
                             dl=b[5] - gc[j, 5], dw=b[4] - gc[j, 4]))
    return pd.DataFrame(rows).set_index("idx").reindex(var.index)


FEATURES = ["log_n_cells", "prob", "prob_max", "det_prob", "fit_frac", "elong", "log_dist"]


def feature_matrix(var):
    f = var.assign(log_n_cells=np.log(var["n_cells"]), log_dist=np.log1p(var["dist"]))
    return f[FEATURES].fillna(0.0).to_numpy(float)


def average_precision(boxes, geometry, var, order_score):
    """AP with ``var``'s boxes appended below the weakest detection of the whole run,
    ordered among themselves by ``order_score`` (global min-max scaled)."""
    floor = float(boxes["score"].min())
    s = np.asarray(order_score, float)
    s = (s - s.min()) / max(s.max() - s.min(), 1e-12) if len(s) else s
    extra = pd.Series(floor * 0.99 * (0.5 + 0.5 * s), index=var.index) if len(var) else None
    stat = {t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in IOUS}
    groups = dict(tuple(var.groupby("frame"))) if len(var) else {}
    for frame, rows in boxes.groupby("frame"):
        gt, pred = geometry[f"gt_{frame}"], geometry[f"pred_{frame}"]
        scores = rows["score"].to_numpy()
        allb, alls = [pred], [scores]
        if frame in groups:
            g = groups[frame]
            allb.append(corners(list(g["box"])))
            alls.append(extra.loc[g.index].to_numpy())
        allb, alls = np.concatenate(allb), np.concatenate(alls)
        _, tp, _ = match_boxes(torch.tensor(allb) if len(allb) else None,
                               torch.tensor(alls) if len(alls) else None,
                               torch.tensor(gt) if len(gt) else None, IOUS)
        for t in IOUS:
            stat[t]["gt"] += len(gt)
            stat[t]["score"].extend(alls.tolist())
            stat[t]["tp"].extend(tp[t].tolist())
            stat[t]["fp"].extend((1 - tp[t]).tolist())
    return {t: calculate_ap(stat, t)[0] for t in IOUS}


def main():
    opt = parse_args()
    if opt.write_selector:
        write_selector(opt)
        return
    out_dir = opt.out_dir or os.path.join(opt.model_dir, "analysis_proposals")
    os.makedirs(out_dir, exist_ok=True)
    fit_boxes, fit_geo, fit_props, fit_meta = load_run(opt.model_dir, opt.fit_prefix, opt.min_cells)
    boxes, geo, props, meta = load_run(opt.model_dir, opt.prefix, opt.min_cells)
    car = detected_sizes(fit_boxes, fit_geo)
    print(f"car size from the fit run's detections (h, w, l): {np.round(car, 2)}; "
          f"proposals: fit {len(fit_props)}, evaluated {len(props)}")
    fit_missed, missed = missed_gt(fit_boxes, fit_geo), missed_gt(boxes, geo)
    n_missed = sum(int(m.sum()) for m in missed.values())

    base = average_precision(boxes, geo, pd.DataFrame(), [])
    print("detections only: " + "  ".join(f"AP{int(t * 100)} {base[t]:.4f}" for t in IOUS))
    diag, aps = [], []
    for name in VARIANTS:
        fv = build_variant(fit_props, name, fit_meta, car)
        ev = build_variant(props, name, meta, car)
        fl = iou_to_gt(fv, fit_geo, fit_missed)
        el = iou_to_gt(ev, geo, missed)
        hit = el["iou"] > 0
        diag.append({
            "variant": name, "boxes": len(ev), "touch a missed car": f"{hit.mean():.1%}",
            "IoU>=0.3": int((el["iou"] >= 0.3).sum()), "IoU>=0.5": int((el["iou"] >= 0.5).sum()),
            "IoU>=0.7": int((el["iou"] >= 0.7).sum()),
            "median IoU | touch": el.loc[hit, "iou"].median(),
            "median |lon| m": el.loc[hit, "lon"].abs().median(),
            "median |lat| m": el.loc[hit, "lat"].abs().median(),
            "median |dyaw| deg": el.loc[hit, "dyaw"].median(),
            "share |dyaw|>45": (el.loc[hit, "dyaw"] > 45).mean(),
            "median dl m": el.loc[hit, "dl"].median(), "median dw m": el.loc[hit, "dw"].median(),
            "touch 2+ missed cars": f"{(el['touch'] >= 2).mean():.1%}",
        })
        y = (fl["iou"] >= 0.5).to_numpy(int)
        orders = {"mask probability": ev["prob"].to_numpy()}
        if len(np.unique(y)) == 2:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(feature_matrix(fv), y)
            orders["learned (fit run)"] = model.predict_proba(feature_matrix(ev))[:, 1]
        for oname, o in orders.items():
            ap = average_precision(boxes, geo, ev, o)
            aps.append({"variant": name, "order": oname,
                        **{f"dAP{int(t * 100)}": ap[t] - base[t] for t in IOUS}})
            print(f"  {name:9s} {oname:18s} " + "  ".join(
                f"AP{int(t * 100)} {ap[t]:.4f} ({ap[t] - base[t]:+.4f})" for t in IOUS), flush=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    d = pd.DataFrame(diag)
    print(f"\n#1 diagnostic on {opt.prefix}: {n_missed} ground-truth cars missed by the detections")
    print(d.round(3).to_string(index=False))
    a = pd.DataFrame(aps)
    print("\nAP change vs detections only (proposals ranked below every detection)")
    print(a.round(4).to_string(index=False))
    tag = opt.prefix.replace("levy_intermediate_", "")
    d.to_csv(os.path.join(out_dir, f"diagnostic_{tag}.csv"), index=False)
    a.to_csv(os.path.join(out_dir, f"ap_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
