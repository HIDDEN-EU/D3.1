"""
Where2comm-style sparse communication for HEAL's pyramid fusion: every
collaborator transmits only a fraction of its BEV cells, chosen from its own
single-agent evidence, before anything is warped into the ego frame.

    call site   PyramidFusion.forward_collab (pyramid_fuse.py) and
                LevyPyramidFusion.forward_collab, inside the per-level loop,
                after the camera crop mask
    built by    PyramidFusion.__init__
    config      model.args.fusion_backbone.comm_mask
                (fusion_backbone.levy.comm_mask is still read by LevyPyramidFusion)
    disable     drop the block, or set `enable: false`

A cell that is not transmitted never reaches the ego: its feature is zeroed AND
its fusion score is set to 0, which is how the fusion already marks "not
observed" (warp padding, camera crop). The ego then keeps the cell to itself
instead of averaging its own feature with a zero. The ego never transmits, so
it is never masked.

Selection (``select``), per agent and per pyramid level, over the cells the
agent would send under full communication (score > 0, i.e. inside its camera
crop):

``evidence``
    The agent's own Levy evidence at this level, Lambda_j = tau * exp(gamma *
    S_j). With one agent this is a monotone function of S_j = sigmoid(occ_j),
    so the ranking -- and hence the mask -- is exactly Where2comm's confidence
    mask on HEAL's own occupancy head, for any tau and gamma.
``evidence_ms``
    The agent's own foreground averaged over all pyramid levels, resampled to
    this level's grid. Not a function of one level's S, so it ranks cells
    differently from ``evidence``. ``ms_space`` sets what is averaged:
    ``prob`` S = sigmoid(occ) in (0, 1); ``logit`` the raw head output occ;
    ``lambda`` the un-normalised intensity exp(ms_gamma * S) -- tau cancels in
    the ranking, and ms_gamma = 1 is exactly the term HEAL's softmax divides by
    its sum over agents. Larger ms_gamma moves the mean towards the max level.
``det_conf``
    Where2comm's own confidence map: the shared detection head run on this
    agent's features alone (decode -> shrink -> cls_head), sigmoid, max over
    anchors. It lives on the level-0 grid; coarser levels take its max over
    each coarse cell. Needs ``set_detector`` (done by HeterPyramidCollab).
``random``
    Uniform over the same cells; the bandwidth-matched control.

``rate`` keeps the top ``rate`` fraction of those cells per agent, so every
selector sends the same number of cells. ``threshold`` (on S, 0..1) instead
keeps every cell whose own probability reaches it (S for ``evidence``, the
head's confidence for ``det_conf``) and reports the rate that results.

Parameter-free: the state_dict is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.LevyHooks.common import announce, note_disabled

SELECTORS = ("evidence", "evidence_ms", "det_conf", "random")


class CommMask(nn.Module):
    def __init__(self, select="evidence", rate=1.0, threshold=None, levels=None,
                 num_levels=3, seed=0, ms_space="prob", ms_gamma=1.0):
        super().__init__()
        assert select in SELECTORS, f"comm_mask.select must be one of {SELECTORS}"
        assert threshold is None or select in ("evidence", "det_conf"), (
            "comm_mask.threshold is only defined for select: evidence / det_conf")
        self.select = select
        assert ms_space in ("prob", "logit", "lambda"), (
            "comm_mask.ms_space must be prob, logit or lambda")
        self.ms_space = ms_space
        self.ms_gamma = float(ms_gamma)
        self.rate = float(rate)
        assert 0.0 <= self.rate <= 1.0, "comm_mask.rate must lie in [0, 1]"
        self.threshold = None if threshold is None else float(threshold)
        self.levels = (set(range(num_levels)) if levels is None
                       else set(int(l) for l in levels))
        self.generator = None
        self.seed = int(seed)
        # Per-forward accounting, one dict per masked level.
        self.stats = []
        self._ms_signal = None
        # det_conf: callable(feature_list) -> per-agent cls logits [N, A, H, W].
        # A plain function, not a submodule, so the state_dict is unchanged.
        self.detector = None

    def set_detector(self, fn):
        self.detector = fn

    def extra_repr(self):
        sel = (f"threshold={self.threshold:g}" if self.threshold is not None
               else f"rate={self.rate:g}")
        if self.select == "evidence_ms":
            sel += f", ms_space={self.ms_space}"
            if self.ms_space == "lambda":
                sel += f", ms_gamma={self.ms_gamma:g}"
        return f"select={self.select}, {sel}, levels={sorted(self.levels)}"

    def begin_forward(self, occ_map_list, feature_list=None):
        """Reset the stats; precompute the cross-level / detector signal if needed."""
        self.stats = []
        self._ms_signal = None
        if self.select == "det_conf":
            assert self.detector is not None, "comm_mask det_conf needs set_detector"
            conf = torch.sigmoid(self.detector(feature_list)).max(dim=1, keepdim=True).values
            self._ms_signal = [
                conf if o.shape[-2:] == conf.shape[-2:]
                else F.adaptive_max_pool2d(conf, o.shape[-2:]) for o in occ_map_list]
        if self.select == "evidence_ms":
            # Mean over levels, on each level's grid, of S, the logit or exp(gamma S).
            if self.ms_space == "prob":
                probs = [torch.sigmoid(o) for o in occ_map_list]
            elif self.ms_space == "logit":
                probs = list(occ_map_list)
            else:
                probs = [torch.exp(self.ms_gamma * torch.sigmoid(o)) for o in occ_map_list]
            self._ms_signal = []
            for target in probs:
                size = target.shape[-2:]
                acc = sum(F.interpolate(p, size=size, mode="bilinear",
                                        align_corners=False)
                          if p.shape[-2:] != size else p for p in probs)
                self._ms_signal.append(acc / len(probs))

    def _signal(self, level, occ_map):
        if self.select == "evidence":
            return occ_map  # any monotone map of the logit ranks the same
        if self.select in ("evidence_ms", "det_conf"):
            return self._ms_signal[level]
        if self.generator is None or self.generator.device != occ_map.device:
            self.generator = torch.Generator(device=occ_map.device)
            self.generator.manual_seed(self.seed)
        return torch.rand(occ_map.shape, generator=self.generator,
                          device=occ_map.device)

    def apply(self, level, features, score, occ_map, record_len):
        """
        Returns ``(features, score)`` with the untransmitted cells of every
        collaborator zeroed in both. ``score`` must already carry the camera
        crop (0 = not observed), so the rate is a fraction of what the agent
        would have sent under full communication.
        """
        if level not in self.levels or (self.threshold is None and self.rate >= 1.0):
            return features, score

        n = features.shape[0]
        is_ego = torch.zeros(n, dtype=torch.bool, device=features.device)
        offset = 0
        for r in record_len:
            is_ego[offset] = True
            offset += int(r)

        observed = score > 0                                  # [N, 1, H, W]
        if self.threshold is not None:
            prob = (self._ms_signal[level] if self.select == "det_conf"
                    else torch.sigmoid(occ_map))
            keep = prob >= self.threshold
        else:
            signal = self._signal(level, occ_map).masked_fill(~observed, -float("inf"))
            flat = signal.flatten(1)                          # [N, HW]
            n_obs = observed.flatten(1).sum(1)                # [N]
            k = torch.floor(n_obs.to(torch.float64) * self.rate).long()
            order = flat.argsort(dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(flat.shape[1], device=flat.device)
                          .expand_as(order))
            keep = (rank < k[:, None]).view_as(observed)
        keep = (keep & observed) | is_ego.view(-1, 1, 1, 1)

        collab = ~is_ego
        if collab.any():
            sent = (keep & observed)[collab].sum().item()
            avail = observed[collab].sum().item()
            self.stats.append({
                "level": level,
                "sent_cells": int(sent),
                "full_cells": int(avail),
                "channels": int(features.shape[1]),
            })

        keep_f = keep.to(features.dtype)
        return features * keep_f, score * keep_f


def build_comm_mask(cfg, num_levels, where="PyramidFusion"):
    """
    Build a :class:`CommMask` from the ``comm_mask`` yaml block, or ``None``.
        comm_mask:
          enable: false
          select: evidence     # evidence | evidence_ms | det_conf | random
          rate: 0.1            # fraction of each collaborator's cells sent
          threshold: null      # probability cut; overrides rate (evidence / det_conf)
          levels: null         # null = every pyramid level
          seed: 0              # for select: random
          ms_space: prob       # evidence_ms averages prob | logit | lambda
          ms_gamma: 1.0        # lambda = exp(ms_gamma * S)
    """
    cfg = cfg or {}
    if not cfg.get("enable", False):
        note_disabled("comm_mask", where, cfg)
        return None
    mask = CommMask(
        select=cfg.get("select", "evidence"),
        rate=float(cfg.get("rate", 1.0)),
        threshold=cfg.get("threshold", None),
        levels=cfg.get("levels", None),
        num_levels=num_levels,
        seed=int(cfg.get("seed", 0)),
        ms_space=cfg.get("ms_space", "prob"),
        ms_gamma=float(cfg.get("ms_gamma", 1.0)),
    )
    return announce("comm_mask", mask, where, mask.extra_repr())
