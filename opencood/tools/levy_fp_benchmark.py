# -*- coding: utf-8 -*-
"""
Levy false-positive benchmark, replayed offline from a single inference run.

Reads the per-box arrays that ``inference.py --levy_diag`` dumps
(``levy_diag_boxes.npz``: sigma, evidence, disagreement, coverage, score,
is_tp@{30,50,70}, boxes-per-frame, gt_total) and runs the blueprint's
Protocol 1 (3-way re-scoring) and Protocol 3 (sparsification / AUSE) without
touching a GPU.

Protocol 1 -- 3-way re-scoring
    score_final = score_det * (1 - exp(-Lambda)) * exp(-beta * Disagreement)

    The evidence factor ``1 - exp(-Lambda)`` saturates to 1 wherever any agent
    reports foreground and collapses toward 0 on cells nobody supports; the
    disagreement factor punishes cells where the agents' features conflict.
    ``--beta_grid`` sweeps beta and reports AP plus false positives per frame at
    fixed recall.

Protocol 3 -- sparsification / AUSE
    Discard a growing fraction of the predictions ordered by each trust signal
    and watch what is left. Two error measures are reported:
      * ``fp_rate``  -- false positives / kept boxes (the measure the existing
        ``levy_error_metrics`` uses).
      * ``1 - AP70`` -- the measure the blueprint asks for.
    AUSE is the area between a signal's curve and the oracle's; ``ause_norm``
    rescales it so 0 = oracle and 1 = random, which is the only version
    comparable across error measures.

CAVEAT on every re-scored / re-ranked AP in this script: the TP/FP labels come
from the matching that ran at inference time in the original score order, and
are held fixed while the scores are permuted. Re-scoring can in principle change
which prediction claims a GT box when two overlap, so these APs are a re-ranking
approximation. Dropping boxes (Protocol 3, and ``levy_drop_sweep.py``) has no
such caveat for the kept set -- removing a box never re-assigns another one's
label -- beyond releasing a GT that a dropped TP had claimed.

Usage
-----
    python opencood/tools/levy_fp_benchmark.py \
        --npz opencood/logs/<run>/levy_intermediate_diag_.../levy_diag_boxes.npz \
        --out_dir opencood/logs/<run>/levy_benchmark
"""
import argparse
import json
import os

import numpy as np

from opencood.tools.levy_metrics import (
    BoxDump, ap_from_labels, auroc, fp_at_recall, max_recall, DUMP_IOUS)

IOUS = DUMP_IOUS


# --------------------------------------------------------------------------- #
# Protocol 3 -- sparsification curves
# --------------------------------------------------------------------------- #
def sparsification_curve(order, is_tp_ref, tp_by_iou, score, gt_total,
                         fractions):
    """
    Discard ``order[:k]`` for a growing k and measure what survives.

    ``order`` lists the boxes worst-first according to some criterion, so a good
    criterion drives the remaining error down fastest.
    """
    n = order.shape[0]
    fp_rate, err_ap70 = [], []
    for f in fractions:
        k = int(round(f * n))
        keep = order[k:]
        if keep.shape[0] == 0:
            fp_rate.append(0.0)
            err_ap70.append(err_ap70[-1] if err_ap70 else 1.0)
            continue
        fp_rate.append(float((~is_tp_ref[keep]).mean()))
        err_ap70.append(1.0 - ap_from_labels(tp_by_iou[0.7][keep],
                                             score[keep], gt_total))
    return np.array(fp_rate), np.array(err_ap70)


def area(curve, fractions):
    return float(np.trapz(curve, fractions))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True,
                   help='levy_diag_boxes.npz from inference.py --levy_diag')
    p.add_argument('--out_dir', default=None,
                   help='where to write JSON + plots (default: next to --npz)')
    p.add_argument('--ref_iou', type=float, default=0.5,
                   help='IoU whose TP/FP labels define a false positive for the '
                        'FP-rate / AUROC columns (default 0.5)')
    p.add_argument('--beta_grid', default='0,0.05,0.1,0.2,0.5,1.0,2.0',
                   help='comma-separated beta values for the 3-way re-scoring')
    p.add_argument('--n_points', type=int, default=51,
                   help='points on each sparsification curve')
    p.add_argument('--max_frac', type=float, default=0.9,
                   help='largest discarded fraction on the curves')
    p.add_argument('--tag', default='',
                   help='label written into the JSON (e.g. the noise level)')
    opt = p.parse_args()

    out_dir = opt.out_dir or os.path.dirname(os.path.abspath(opt.npz))
    os.makedirs(out_dir, exist_ok=True)

    dump = BoxDump(opt.npz)
    sigma = dump.sigma
    score = dump.score
    evidence = dump.evidence
    disagreement = dump.disagreement
    coverage = dump.coverage
    gt_total = dump.gt_total
    n_frames = dump.n_frames
    tp_by_iou = dump.tp_by_iou
    ref = dump.labels(opt.ref_iou)
    is_fp = ~ref
    n = sigma.shape[0]

    report = {'npz': os.path.abspath(opt.npz), 'tag': opt.tag,
              'n_boxes': n, 'n_frames': n_frames, 'gt_total': gt_total,
              'ref_iou': opt.ref_iou,
              'n_tp': int(ref.sum()), 'n_fp': int(is_fp.sum())}

    # ---------------- baseline ------------------------------------------- #
    base_ap = {f'{iou:g}': ap_from_labels(tp_by_iou[iou], score, gt_total)
               for iou in IOUS}
    base_fp_rate = float(is_fp.mean())
    base_fp_frame = fp_at_recall(ref, score, gt_total, n_frames)
    report['baseline'] = {'ap': base_ap, 'fp_rate': base_fp_rate,
                          'fp_per_frame_at_recall': base_fp_frame,
                          'max_recall': max_recall(ref, gt_total),
                          'boxes_per_frame': n / max(n_frames, 1)}

    print(f"\n=== Levy FP benchmark {('[' + opt.tag + ']') if opt.tag else ''} ===")
    print(f"{n} boxes over {n_frames} frames, {gt_total} GT; "
          f"{int(ref.sum())} TP / {int(is_fp.sum())} FP @ IoU {opt.ref_iou:g} "
          f"(FP rate {base_fp_rate:.2%})")
    print(f"baseline AP  .30 {base_ap['0.3']:.4f}  .50 {base_ap['0.5']:.4f}  "
          f".70 {base_ap['0.7']:.4f}")
    print(f"baseline max achievable recall {report['baseline']['max_recall']:.3f}"
          f"  (FP/frame is nan for any level above it)")
    print("baseline FP/frame at recall  "
          + "  ".join(f"{k} -> {v:.3f}" for k, v in base_fp_frame.items()))

    # ---------------- signal descriptives -------------------------------- #
    print(f"\n-- trust signals, TP vs FP means (IoU {opt.ref_iou:g}) --")
    print(f"  {'signal':<14}{'mean|TP':>10}{'mean|FP':>10}{'AUROC->FP':>11}")
    signals = {'sigma': sigma, 'evidence': evidence,
               'disagreement': disagreement, 'coverage': coverage,
               'det_score': score}
    report['signals'] = {}
    for name, v in signals.items():
        a = auroc(v, is_fp)          # >0.5: high value marks a false positive
        report['signals'][name] = {
            'mean_tp': float(v[ref].mean()) if ref.any() else float('nan'),
            'mean_fp': float(v[is_fp].mean()) if is_fp.any() else float('nan'),
            'auroc_fp': a,
            'pct': {f'p{q}': float(np.percentile(v, q))
                    for q in (1, 5, 10, 25, 50, 75, 90, 99)},
        }
        r = report['signals'][name]
        print(f"  {name:<14}{r['mean_tp']:>10.4f}{r['mean_fp']:>10.4f}"
              f"{a:>11.4f}")
    print("  AUROC>0.5: high values mark false positives (usable as-is).")
    print("  AUROC<0.5: signal is inverted -- LOW values mark false positives.")

    # ---------------- conditional value over det_score --------------------- #
    # The control the blueprint does not ask for but that decides whether any of
    # this is worth shipping. Every trust signal correlates with the detector's
    # own confidence, so a signal can look predictive of error while carrying
    # nothing the score did not already say. Bin the boxes by score and re-run
    # the AUROC inside each bin: if it collapses to ~0.5 the signal is redundant.
    print(f"\n-- conditional AUROC within det_score deciles "
          f"(0.5 = redundant given confidence) --")
    edges = np.unique(np.quantile(score, np.linspace(0, 1, 11)))
    bin_idx = np.clip(np.searchsorted(edges, score, side='right') - 1,
                      0, len(edges) - 2)
    report['conditional_auroc'] = {}

    def cond_auroc(v):
        """AUROC of v against FP, averaged over det_score deciles."""
        num = den = 0.0
        used = 0
        per_bin = []
        for b in range(len(edges) - 1):
            m = bin_idx == b
            if m.sum() < 50 or is_fp[m].sum() == 0 or (~is_fp[m]).sum() == 0:
                per_bin.append(None)
                continue
            a = auroc(v[m], is_fp[m])
            per_bin.append(a)
            num += a * m.sum()
            den += m.sum()
            used += 1
        return (num / den if den else float('nan')), per_bin, used

    print(f"  {'signal':<14}{'weighted mean AUROC|score':>26}{'n_bins_used':>13}")
    for name in ('sigma', 'evidence', 'disagreement', 'coverage'):
        wm, per_bin, used = cond_auroc(signals[name])
        report['conditional_auroc'][name] = {
            'weighted_mean': wm, 'per_bin': per_bin,
            'marginal': report['signals'][name]['auroc_fp']}
        print(f"  {name:<14}{wm:>26.4f}{used:>13d}")
    print("  Compare with the marginal AUROC above: a signal that is strongly "
          "inverted\n  marginally but sits at ~0.5 here only re-expresses the "
          "detector's confidence.")

    # ---------------- alternative trust-map parameterisations -------------- #
    # Channels the model computed alongside the base map in the same forward pass:
    # cosine disagreement (angular conflict, magnitude-invariant) and shifted /
    # re-tempered intensity (Lambda moved into the band where phi and the gate
    # respond). Same detections, same boxes -- only the trust map differs.
    var_keys = sorted(dump.variants)
    report['variants'] = {}
    if var_keys:
        print(f"\n-- alternative trust-map parameterisations "
              f"({len(var_keys)} channels) --")
        hdr = (f"  {'channel':<26}{'mean|TP':>10}{'mean|FP':>10}"
               f"{'AUROC':>9}{'AUROC|score':>13}")
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for label in var_keys:
            v = dump.variants[label]
            if v.shape[0] != n:
                continue
            a = auroc(v, is_fp)
            ca, _, _ = cond_auroc(v)
            report['variants'][label] = {
                'mean_tp': float(v[ref].mean()) if ref.any() else float('nan'),
                'mean_fp': float(v[is_fp].mean()) if is_fp.any() else float('nan'),
                'auroc_fp': a, 'cond_auroc_fp': ca,
                'pct': {f'p{q}': float(np.percentile(v, q))
                        for q in (1, 10, 50, 90, 99)},
            }
            r = report['variants'][label]
            print(f"  {label:<26}{r['mean_tp']:>10.4f}{r['mean_fp']:>10.4f}"
                  f"{a:>9.4f}{ca:>13.4f}")
        base_d = report['signals']['disagreement']
        print(f"  {'(base) disagreement':<26}{base_d['mean_tp']:>10.4f}"
              f"{base_d['mean_fp']:>10.4f}{base_d['auroc_fp']:>9.4f}"
              f"{report['conditional_auroc']['disagreement']['weighted_mean']:>13.4f}")

    # ---------------- Protocol 1: 3-way re-scoring ------------------------ #
    print(f"\n-- Protocol 1: 3-way re-scoring "
          f"score*(1-exp(-Lambda))*exp(-beta*Disagreement) --")
    ev_factor = 1.0 - np.exp(-evidence)
    betas = [float(x) for x in opt.beta_grid.split(',') if x.strip()]
    hdr = (f"  {'beta':>6} {'AP.30':>8} {'AP.50':>8} {'AP.70':>8} "
           f"{'dAP.70':>8} {'FP/fr@.8':>9} {'FP/fr@.9':>9}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    report['protocol1_rescore'] = []
    for b in betas:
        new_score = score * ev_factor * np.exp(-b * disagreement)
        aps = {f'{iou:g}': ap_from_labels(tp_by_iou[iou], new_score, gt_total)
               for iou in IOUS}
        fpf = fp_at_recall(ref, new_score, gt_total, n_frames)
        report['protocol1_rescore'].append(
            {'beta': b, 'ap': aps, 'fp_per_frame_at_recall': fpf})
        print(f"  {b:>6g} {aps['0.3']:>8.4f} {aps['0.5']:>8.4f} "
              f"{aps['0.7']:>8.4f} {aps['0.7'] - base_ap['0.7']:>+8.4f} "
              f"{fpf['0.8']:>9.3f} {fpf['0.9']:>9.3f}")
    print(f"  (baseline {base_ap['0.3']:.4f} {base_ap['0.5']:.4f} "
          f"{base_ap['0.7']:.4f}      "
          f"{base_fp_frame['0.8']:.3f}     {base_fp_frame['0.9']:.3f})")

    # Evidence-only variant, with a strength exponent. Disagreement is the half
    # of the blueprint formula that misfires here (it is inverted, so exp(-beta*D)
    # penalises the correct boxes), so the useful family is beta = 0 with the
    # evidence gate sharpened: score * (1 - exp(-Lambda))^k.
    print(f"\n-- Protocol 1b: evidence-only re-scoring "
          f"score*(1-exp(-Lambda))^k --")
    hdr = (f"  {'k':>6} {'AP.30':>8} {'AP.50':>8} {'AP.70':>8} "
           f"{'dAP.70':>8} {'FP/fr@.8':>9} {'FP/fr@.9':>9}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    report['protocol1b_evidence_only'] = []
    for k in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        new_score = score * np.power(ev_factor, k)
        aps = {f'{iou:g}': ap_from_labels(tp_by_iou[iou], new_score, gt_total)
               for iou in IOUS}
        fpf = fp_at_recall(ref, new_score, gt_total, n_frames)
        report['protocol1b_evidence_only'].append(
            {'k': k, 'ap': aps, 'fp_per_frame_at_recall': fpf})
        print(f"  {k:>6g} {aps['0.3']:>8.4f} {aps['0.5']:>8.4f} "
              f"{aps['0.7']:>8.4f} {aps['0.7'] - base_ap['0.7']:>+8.4f} "
              f"{fpf['0.8']:>9.3f} {fpf['0.9']:>9.3f}")
    print(f"  (baseline {base_ap['0.3']:.4f} {base_ap['0.5']:.4f} "
          f"{base_ap['0.7']:.4f}      "
          f"{base_fp_frame['0.8']:.3f}     {base_fp_frame['0.9']:.3f})")

    # ---------------- 3-way re-scoring with each disagreement channel ------- #
    # The blueprint's penalty assumes high tr Sigma marks a false positive. If a
    # channel is inverted the correct sign is the other one, so beta is swept over
    # both signs and the best is reported per channel -- a channel that only helps
    # at beta < 0 is telling you its sign is backwards, not that it works.
    dis_channels = {'(base) disagreement': disagreement}
    for label in var_keys:
        if label.endswith('.disagreement'):
            v = dump.variants[label]
            if v.shape[0] == n:
                dis_channels[label] = v
    if len(dis_channels) > 1:
        print(f"\n-- 3-way re-scoring per disagreement channel: "
              f"score*(1-exp(-Lambda))*exp(-beta*D) --")
        hdr = (f"  {'channel':<26}{'best beta':>10}{'AP.70':>9}{'dAP.70':>9}"
               f"{'FP/fr@.8':>10}")
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        report['rescore_by_channel'] = {}
        betas2 = [-2.0, -1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
        for label, v in dis_channels.items():
            # normalise scale so one beta grid is comparable across channels
            sc = np.percentile(v, 90)
            vv = v / sc if sc > 0 else v
            rows = []
            for b in betas2:
                ns = score * ev_factor * np.exp(-b * vv)
                rows.append({'beta': b,
                             'ap70': ap_from_labels(tp_by_iou[0.7], ns, gt_total),
                             'fp8': fp_at_recall(ref, ns, gt_total,
                                                 n_frames)['0.8']})
            bestr = max(rows, key=lambda r: r['ap70'])
            report['rescore_by_channel'][label] = {'rows': rows, 'best': bestr,
                                                   'p90_scale': float(sc)}
            print(f"  {label:<26}{bestr['beta']:>10g}{bestr['ap70']:>9.4f}"
                  f"{bestr['ap70'] - base_ap['0.7']:>+9.4f}"
                  f"{bestr['fp8']:>10.3f}")
        print(f"  (baseline AP70 {base_ap['0.7']:.4f}, "
              f"FP/frame@.8 {base_fp_frame['0.8']:.3f}; D scaled by its own p90)")

    # ---------------- evidence gate, scale x k ----------------------------- #
    # A logit shift b is exactly a rescale of Lambda by exp(gamma*b), so it cannot
    # reorder boxes -- but it does move where (1-exp(-Lambda))^k sits on its curve,
    # and that is a free parameter worth sweeping offline.
    print(f"\n-- evidence gate: score * (1-exp(-c*Lambda))^k --")
    scales = [1.0, 2.0, 4.48, 10.0]     # 4.48 = exp(0.25*6), i.e. logit shift +6
    ks = [0.5, 1.0, 2.0]
    hdr = (f"  {'c':>6} {'shift b':>8} {'k':>5} {'AP.70':>8} {'dAP.70':>9} "
           f"{'FP/fr@.8':>9} {'FP/fr@.5':>9}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    report['evidence_gate_grid'] = []
    best = None
    for c in scales:
        for k in ks:
            fac = np.power(1.0 - np.exp(-c * evidence), k)
            ns = score * fac
            ap70 = ap_from_labels(tp_by_iou[0.7], ns, gt_total)
            fpf = fp_at_recall(ref, ns, gt_total, n_frames)
            row = {'scale': c, 'shift_at_gamma_0.25': float(np.log(c) / 0.25),
                   'k': k, 'ap70': ap70, 'fp_per_frame_at_recall': fpf}
            report['evidence_gate_grid'].append(row)
            print(f"  {c:>6g} {np.log(c)/0.25:>8.2f} {k:>5g} {ap70:>8.4f} "
                  f"{ap70 - base_ap['0.7']:>+9.4f} "
                  f"{fpf['0.8']:>9.3f} {fpf['0.5']:>9.3f}")
            cand = fpf['0.8'] if fpf['0.8'] == fpf['0.8'] else fpf['0.5']
            if ap70 >= base_ap['0.7'] and (best is None or cand < best[0]):
                best = (cand, row)
    print(f"  (baseline {base_ap['0.7']:.4f}            "
          f"{base_fp_frame['0.8']:.3f}     {base_fp_frame['0.5']:.3f})")
    if best:
        b = best[1]
        print(f"  best non-regressing: c={b['scale']:g} k={b['k']:g} -> "
              f"AP70 {b['ap70']:.4f}, FP/frame {best[0]:.3f}")
        report['evidence_gate_best'] = b

    # ---------------- Protocol 3: sparsification / AUSE ------------------- #
    fractions = np.linspace(0.0, opt.max_frac, opt.n_points)
    rng = np.random.default_rng(0)

    # Worst-first orderings. A signal whose AUROC is below 0.5 is inverted, so
    # its worst-first order is ascending; using the wrong direction would be
    # measuring the signal upside down.
    # One oracle per error measure. An oracle built from IoU-0.5 labels is not
    # the lower bound for a 1-AP70 curve -- it leaves IoU-0.7 false positives in
    # place -- and scoring against the wrong one lets an ordinary signal appear
    # to beat "perfect", which is how a negative AUSE shows up.
    oracle_by_measure = {
        'fp_rate': np.argsort(-is_fp.astype(np.float64), kind='mergesort'),
        'err_ap70': np.argsort(-(~tp_by_iou[0.7]).astype(np.float64),
                               kind='mergesort'),
    }
    orders = {
        'oracle': oracle_by_measure['fp_rate'],   # curve shown for fp_rate
        'random': rng.permutation(n),
        'det_score(asc)': np.argsort(score, kind='mergesort'),
    }
    for name in ('sigma', 'evidence', 'disagreement', 'coverage'):
        v = signals[name]
        a = report['signals'][name]['auroc_fp']
        asc = a < 0.5                       # low values mark FPs -> drop lowest
        orders[f'{name}({"asc" if asc else "desc"})'] = \
            np.argsort(v if asc else -v, kind='mergesort')

    curves, areas = {}, {}
    for name, order in orders.items():
        fpr_c, ap_c = sparsification_curve(order, ref, tp_by_iou, score,
                                           gt_total, fractions)
        curves[name] = {'fp_rate': fpr_c.tolist(), 'err_ap70': ap_c.tolist()}
        areas[name] = {'fp_rate': area(fpr_c, fractions),
                       'err_ap70': area(ap_c, fractions)}

    # Oracle areas, each from the ordering that is actually optimal for its own
    # error measure.
    oracle_area = {}
    for measure in ('fp_rate', 'err_ap70'):
        o_fpr, o_ap = sparsification_curve(oracle_by_measure[measure], ref,
                                           tp_by_iou, score, gt_total, fractions)
        oracle_area[measure] = area(o_fpr if measure == 'fp_rate' else o_ap,
                                    fractions)
        if measure == 'err_ap70':
            curves['oracle']['err_ap70'] = o_ap.tolist()
    areas['oracle'] = dict(oracle_area)

    report['protocol3'] = {'fractions': fractions.tolist(),
                           'curves': curves, 'areas': areas,
                           'oracle_area': oracle_area, 'ause': {}}

    print(f"\n-- Protocol 3: sparsification / AUSE (discard up to "
          f"{opt.max_frac:.0%}, worst-first) --")
    hdr = (f"  {'ordering':<20}{'AUSE(fp_rate)':>15}{'norm':>8}"
           f"{'AUSE(1-AP70)':>15}{'norm':>8}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for name in orders:
        row = {}
        for measure in ('fp_rate', 'err_ap70'):
            a_m = areas[name][measure]
            a_o = oracle_area[measure]
            a_r = areas['random'][measure]
            ause = a_m - a_o
            denom = a_r - a_o
            row[measure] = {
                'ause': ause,
                'ause_norm': ause / denom if abs(denom) > 1e-12 else float('nan'),
            }
        report['protocol3']['ause'][name] = row
        print(f"  {name:<20}{row['fp_rate']['ause']:>15.5f}"
              f"{row['fp_rate']['ause_norm']:>8.3f}"
              f"{row['err_ap70']['ause']:>15.5f}"
              f"{row['err_ap70']['ause_norm']:>8.3f}")
    print("  ause_norm: 0 = oracle, 1 = random. Below 1 means the signal beats "
          "chance at\n  finding false positives; above 1 means it is worse than "
          "random.")

    # ---------------- write out ------------------------------------------ #
    tag = f"_{opt.tag}" if opt.tag else ""
    json_path = os.path.join(out_dir, f'levy_fp_benchmark{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {json_path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, measure, ylabel in (
                (axes[0], 'fp_rate', 'FP rate of kept boxes'),
                (axes[1], 'err_ap70', r'$1 - AP_{70}$ of kept boxes')):
            for name in orders:
                style = '--' if name in ('oracle', 'random') else '-'
                lw = 1.4 if name in ('oracle', 'random') else 2.0
                ax.plot(fractions, curves[name][measure], style, lw=lw,
                        label=name)
            ax.set_xlabel('fraction of predictions discarded')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle('Levy sparsification curves'
                     + (f' -- {opt.tag}' if opt.tag else ''))
        fig.tight_layout()
        plot_path = os.path.join(out_dir, f'levy_sparsification{tag}.png')
        fig.savefig(plot_path, dpi=140)
        print(f"plot   -> {plot_path}")
    except Exception as e:                                  # pragma: no cover
        print(f"[warn] plotting failed: {e}")


if __name__ == '__main__':
    main()
