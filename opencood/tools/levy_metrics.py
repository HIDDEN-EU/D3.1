# -*- coding: utf-8 -*-
"""
Shared metric primitives for the Levy false-positive analysis.

This is the leaf module of the Levy tooling: it imports only numpy and
``voc_ap``, and everything else imports from here. It exists so that the AP,
AUROC and operating-point definitions used by the benchmark harness, the drop
sweep and the inference-time diagnostic are literally the same code -- an
earlier revision had three separate copies of the AP calculation and two of the
tie-corrected AUROC, which is exactly the kind of drift that makes two tables in
one report disagree.

The per-box dump these functions consume is written by
``inference.py --levy_diag`` as ``levy_diag_boxes.npz``. Its schema:

======================  =========================================================
``sigma``               (N,) per-box sigma_hat, pooled over the box footprint
``evidence``            (N,) per-box Lambda
``disagreement``        (N,) per-box tr Sigma
``coverage``            (N,) agent count at the box centre
``score``               (N,) detector confidence, before any re-scoring
``is_tp_030/050/070``   (N,) bool, matched to a GT box at that IoU
``n_boxes_per_frame``   (F,) boxes emitted per frame; ``F`` is the frame count
``gt_total``            scalar, ground-truth boxes over the split
``var__<name>.<chan>``  (N,) the same channel under an alternative trust-map
                        parameterisation (``levy.uncertainty_variants``)
======================  =========================================================

Every array is in raw prediction order, so a column can be paired with any other
by index and the whole parameter search replays on CPU.
"""
import numpy as np

from opencood.utils.eval_utils import voc_ap

#: IoU thresholds the dump carries TP/FP labels for.
DUMP_IOUS = (0.3, 0.5, 0.7)

#: Recall levels the operating-point metric reports by default. The low ones
#: matter because under heavy pose noise the detector never reaches 0.8 at all.
DEFAULT_RECALL_LEVELS = (0.5, 0.6, 0.8, 0.9)


# --------------------------------------------------------------------------- #
# ranking statistics
# --------------------------------------------------------------------------- #
def average_ranks(values):
    """
    Ranks of ``values``, averaged within tied groups.

    Tie handling is not optional here: sigma_hat is structurally 0 wherever
    fewer than two agents overlap, which on OPV2V-H is ~6% of boxes in one
    discrete lump. Ranking that lump arbitrarily biases AUROC.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auroc(score, positive):
    """
    P(score of a random positive > score of a random negative), ties at 0.5.

    Used throughout with ``positive = is_false_positive``, so the reading is:
    above 0.5 the signal marks false positives and can be thresholded as-is;
    0.5 means it carries no information; below 0.5 it is *inverted* and marks
    the boxes that are correct.
    """
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int(positive.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = average_ranks(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# detection metrics from per-box labels
# --------------------------------------------------------------------------- #
def ap_from_labels(is_tp, score, gt_total):
    """
    VOC-2010 AP from per-box TP/FP labels and scores.

    Mirrors :func:`opencood.utils.eval_utils.calculate_ap` but takes the labels
    directly, so a re-scoring or a drop can be evaluated without re-running the
    matching.

    Note the standing caveat for *re-scored* AP: the labels were assigned by the
    matching that ran in the original score order and are held fixed while the
    scores are permuted, so a re-ranked AP is approximate wherever two
    predictions contest the same ground-truth box. Dropping boxes carries no
    such caveat for the kept set.
    """
    score = np.asarray(score, dtype=np.float64)
    if score.shape[0] == 0 or gt_total == 0:
        return 0.0
    order = np.argsort(-score, kind='mergesort')
    tp = np.asarray(is_tp, dtype=bool)[order].astype(np.float64)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(1.0 - tp)
    rec = (tp_c / gt_total).tolist()
    prec = (tp_c / np.maximum(tp_c + fp_c, 1e-12)).tolist()
    ap, _, _ = voc_ap(rec, prec)
    return float(ap)


def fp_at_recall(is_tp, score, gt_total, n_frames,
                 recall_levels=DEFAULT_RECALL_LEVELS):
    """
    False positives per frame at fixed recall.

    Walks the score-ordered predictions to the first point where cumulative
    recall reaches each level and counts the false positives accepted to get
    there. This is the operating point a deployed detector sits at, and it is
    where false-positive suppression either pays or does not -- AP averages over
    every threshold and hides it.

    A level above the achievable recall yields ``nan``; see :func:`max_recall`.
    """
    score = np.asarray(score, dtype=np.float64)
    out = {f'{r:g}': float('nan') for r in recall_levels}
    if score.shape[0] == 0 or gt_total == 0:
        return out
    order = np.argsort(-score, kind='mergesort')
    tp = np.asarray(is_tp, dtype=bool)[order].astype(np.float64)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(1.0 - tp)
    rec = tp_c / gt_total
    for r in recall_levels:
        idx = np.searchsorted(rec, r, side='left')
        if idx < rec.shape[0]:
            out[f'{r:g}'] = float(fp_c[idx] / max(n_frames, 1))
    return out


def max_recall(is_tp, gt_total):
    """Highest recall the predictions can reach, i.e. all true positives kept."""
    return float(np.asarray(is_tp, dtype=bool).sum() / gt_total) \
        if gt_total else float('nan')


# --------------------------------------------------------------------------- #
# the per-box dump
# --------------------------------------------------------------------------- #
class BoxDump:
    """
    ``levy_diag_boxes.npz`` with its columns named.

    Attributes mirror the schema in this module's docstring. ``variants`` maps
    ``'<name>.<channel>'`` to its array, with the ``var__`` prefix stripped.
    """

    def __init__(self, path):
        d = np.load(path)
        self.path = path
        self.raw = d
        self.sigma = d['sigma'].astype(np.float64)
        self.score = d['score'].astype(np.float64)
        self.gt_total = int(d['gt_total'])
        self.tp_by_iou = {iou: d[f'is_tp_{int(round(iou * 100)):03d}'].astype(bool)
                          for iou in DUMP_IOUS}
        # Columns added after the first revision of the dump; a dump written by
        # an older inference.py simply will not have them.
        for name in ('evidence', 'disagreement', 'coverage'):
            setattr(self, name,
                    d[name].astype(np.float64) if name in d.files else None)
        self.n_frames = (int(d['n_boxes_per_frame'].shape[0])
                         if 'n_boxes_per_frame' in d.files else 0)
        self.variants = {k[len('var__'):]: d[k].astype(np.float64)
                         for k in d.files if k.startswith('var__')}
        self.n_boxes = self.sigma.shape[0]

    def channel(self, name):
        """
        Look up a per-box column by name, base or variant.

        Accepts ``'sigma'``, ``'evidence'``, ``'disagreement'``, ``'coverage'``,
        ``'score'``, or a variant channel as ``'cos.disagreement'`` (the
        ``var__`` prefix is optional).
        """
        if name in self.variants:
            return self.variants[name]
        if name.startswith('var__') and name[len('var__'):] in self.variants:
            return self.variants[name[len('var__'):]]
        v = getattr(self, name, None)
        if v is None:
            raise KeyError(
                f"channel {name!r} not in {self.path}. Available: "
                + ", ".join(['sigma', 'evidence', 'disagreement', 'coverage',
                             'score'] + sorted(self.variants)))
        return v

    def labels(self, iou=0.5):
        return self.tp_by_iou[iou]

    def is_fp(self, iou=0.5):
        return ~self.tp_by_iou[iou]
