#!/usr/bin/env python3
"""
demo_UC2_fusion.py - run the UC2 LiDAR + UWB fusion on the synthetic scenario.

    python demo_UC2_fusion.py                       # one run, UWB at 10 Hz
    python demo_UC2_fusion.py --uwb-hz 1 --plot out.png
    python demo_UC2_fusion.py --sweep               # UWB rate 1/2/5/10 Hz x 5 seeds, mean table
    python demo_UC2_fusion.py --dump-dir examples   # example input/output messages as JSON
"""

import argparse
import json
import os

import numpy as np

from evaluate import run, summarize
from UC2_fusion import FusionConfig
from uwb_sim import ScenarioConfig, generate


def report(s: dict) -> str:
    f = lambda v: "  n/a" if not np.isfinite(v) else f"{v:5.2f}"
    return "\n".join([
        f"{'':28s}{'fused':>8s}{'UWB only':>10s}{'LiDAR only':>12s}",
        f"{'RMSE all [m]':28s}{f(s['fused_rmse']):>8s}{f(s['uwb_rmse']):>10s}{f(s['lidar_rmse']):>12s}",
        f"{'RMSE LiDAR occluded [m]':28s}{f(s['fused_rmse_occl']):>8s}{f(s['uwb_rmse_occl']):>10s}{'-':>12s}",
        f"{'RMSE UWB dropout [m]':28s}{f(s['fused_rmse_uwbdown']):>8s}{f(s['uwb_rmse_uwbdown']):>10s}"
        f"{f(s['lidar_rmse_uwbdown']):>12s}",
        f"{'availability':28s}{s['fused_avail']:8.2f}{s['uwb_avail']:10.2f}{s['lidar_avail']:12.2f}",
        f"{'availability, occluded':28s}{s['fused_avail_occl']:8.2f}{'':>10s}{s['lidar_avail_occl']:12.2f}",
        f"fused id changes: {int(s['id_changes'])}   LiDAR link correct: {100 * s['link_correct']:.1f} % "
        f"(linked {100 * s['linked_frac']:.0f} % of the cycles)   duplicate boxes: {100 * s['ghost_frac']:.1f} % "
        f"of the cycles   UWB outliers rejected: {int(s['uwb_outliers'])}"])


def plot(sc, r, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from UC2_fusion import rot2

    t = r.t - sc.t_start
    err = lambda e: np.linalg.norm(e - r.gt, axis=1)
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.45, 3], hspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    for name, e, c in (("UWB only (last sample)", r.uwb_hold, "tab:orange"), ("LiDAR only (oracle box)", r.lidar_hold, "tab:blue"),
                       ("fused", r.fused, "k")):
        ax.plot(t, err(e), color=c, lw=1.6 if name == "fused" else 1.0, label=name)
    for a, b in sc.cfg.vru_occlusions:
        ax.axvspan(a, b, color="tab:blue", alpha=0.12, label="LiDAR occlusion")
    for a, b in sc.cfg.uwb_dropouts:
        ax.axvspan(a, b, color="tab:orange", alpha=0.15, label="UWB dropout")
    if sc.cfg.swap_time:
        ax.axvline(sc.cfg.swap_time, color="tab:red", ls=":", label="LiDAR id swap with neighbour")
    ax.set_ylim(0, 2.5)
    ax.set_ylabel("position error [m]")
    ax.set_xlabel("time [s]")
    ax.set_title(title)
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), ncol=3, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    st = fig.add_subplot(gs[1, :], sharex=ax)
    colors = {"fused": "tab:green", "lidar_only": "tab:blue", "uwb_only": "tab:orange", "coasting": "tab:red", None: "w"}
    for k in range(len(t)):
        st.axvspan(t[k] - 0.05, t[k] + 0.05, color=colors[r.status[k]])
    st.set_yticks([])
    st.set_ylabel("status", rotation=0, labelpad=28)
    from matplotlib.patches import Patch
    st.legend(handles=[Patch(color=colors[n], label=n) for n in ("fused", "lidar_only", "uwb_only", "coasting")],
              ncol=4, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False)

    # trajectory in the (simulated) world frame
    tp = fig.add_subplot(gs[2, 0])
    to_map = lambda k, p: sc.ego_pose(r.t[k])[0] + rot2(sc.ego_pose(r.t[k])[1]) @ p
    for name, est, c, m in (("ground truth", r.gt, "0.6", "-"), ("fused", r.fused, "k", "."),
                            ("UWB (arrived)", r.uwb_hold, "tab:orange", "x")):
        pts = np.array([to_map(k, est[k]) for k in range(len(t))])
        tp.plot(pts[:, 0], pts[:, 1], m, color=c, ms=3, lw=2 if name == "ground truth" else 0.8, label=name)
    tp.set_aspect("equal")
    tp.set_xlabel("East [m]")
    tp.set_ylabel("North [m]")
    tp.legend(fontsize=8)
    tp.grid(alpha=0.3)

    hp = fig.add_subplot(gs[2, 1])
    e = err(r.fused)
    hp.hist([err(r.uwb_hold)[np.isfinite(err(r.uwb_hold))], e[np.isfinite(e)]], bins=np.linspace(0, 2, 41),
            label=["UWB only", "fused"], color=["tab:orange", "k"])
    hp.set_xlabel("position error [m]")
    hp.set_ylabel("cycles")
    hp.legend(fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uwb-hz", type=float, default=10.0)
    ap.add_argument("--lidar-hz", type=float, default=10.0)
    ap.add_argument("--out-hz", type=float, default=10.0, help="publication rate of the fused message")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-swap", action="store_true", help="do not swap the LiDAR ids of the VRU and its neighbour")
    ap.add_argument("--sweep", action="store_true", help="UWB 1/2/5/10 Hz, 5 seeds each, mean table")
    ap.add_argument("--plot", help="save a figure (png)")
    ap.add_argument("--dump-dir", help="write example messages (JSON) there")
    a = ap.parse_args()

    def scenario(hz, seed):
        return generate(ScenarioConfig(uwb_hz=hz, lidar_hz=a.lidar_hz, seed=seed,
                                       swap_time=None if a.no_swap else 14.0))

    if a.sweep:
        keys = ["fused_rmse", "uwb_rmse", "lidar_rmse", "fused_rmse_occl", "uwb_rmse_occl", "fused_rmse_uwbdown",
                "fused_avail", "uwb_avail", "lidar_avail", "id_changes", "link_correct", "ghost_frac"]
        print("mean of 5 seeds; RMSE in metres (ego frame at publication time)\n")
        print(f"{'UWB Hz':>6s} | {'RMSE fused':>10s} {'UWB only':>9s} {'LiDAR only':>10s} | {'occluded: fused':>15s} "
              f"{'UWB only':>9s} | {'UWB dropout: fused':>18s} | {'avail fused':>11s} {'LiDAR':>6s} | "
              f"{'id chg':>6s} {'link ok':>7s} {'dup':>5s}")
        for hz in (1, 2, 5, 10):
            runs = [summarize(run(scenario(hz, s), out_hz=a.out_hz)) for s in range(5)]
            m = {k: np.nanmean([r[k] for r in runs]) for k in keys}
            print(f"{hz:6d} | {m['fused_rmse']:10.3f} {m['uwb_rmse']:9.3f} {m['lidar_rmse']:10.3f} | "
                  f"{m['fused_rmse_occl']:15.3f} {m['uwb_rmse_occl']:9.3f} | {m['fused_rmse_uwbdown']:18.3f} | "
                  f"{m['fused_avail']:11.2f} {m['lidar_avail']:6.2f} | {m['id_changes']:6.1f} "
                  f"{100 * m['link_correct']:6.1f}% {100 * m['ghost_frac']:4.1f}%")
        return

    sc = scenario(a.uwb_hz, a.seed)
    r = run(sc, FusionConfig(), out_hz=a.out_hz, keep_outputs=bool(a.dump_dir))
    print(f"UWB {a.uwb_hz:g} Hz, LiDAR {a.lidar_hz:g} Hz, output {a.out_hz:g} Hz, seed {a.seed}; "
          f"{len(sc.events)} messages\n")
    print(report(summarize(r)))
    if a.plot:
        plot(sc, r, a.plot, f"UC2 fusion, UWB {a.uwb_hz:g} Hz, LiDAR {a.lidar_hz:g} Hz (synthetic scenario, seed {a.seed})")
        print("figure:", a.plot)
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        uwb = next(m for _, m in sc.events if m["metadata"]["sender_id"] == "DENSO_uwb_ODT")
        lid = next(m for _, m in sc.events if m["metadata"]["sender_id"] == "CTAG_lidar_ODTH" and
                   m["metadata"]["sequence_number"] == 40)
        out = next(o for o, t in zip(r.outputs, r.t) if t - sc.t_start >= 3.0)
        for name, obj in (("example_uwb_message.json", uwb), ("example_lidar_message.json", lid),
                          ("example_fused_output.json", out)):
            with open(os.path.join(a.dump_dir, name), "w") as f:
                json.dump(obj, f, indent=2)
        print("examples written to", a.dump_dir)


if __name__ == "__main__":
    main()
