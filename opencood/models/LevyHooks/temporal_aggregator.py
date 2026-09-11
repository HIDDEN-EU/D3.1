"""
Hook 3 -- LevyTemporalAggregator (blueprint Sec. 4).

Continuous-time Levy aggregation over an agent's own historical BEV frames.

    call site   none -- see below
    config      none yet
    disable     n/a

**This hook is not wired into anything, deliberately.** HEAL's OPV2V pipeline is
single-frame: there is no history tensor in the dataloader to feed it. It is also
not zero-retraining -- ``temporal_scorer`` is a fresh 1x1 conv with no pretrained
weights. Adopting it means extending the dataset to stack ``T`` past BEV maps and
then training, which is a larger commitment than the other three hooks.

It lives here complete and self-tested so the blueprint's Sec. 4 is implemented
rather than skipped, and so the work is already done if a temporal dataloader
arrives.
"""

import torch.nn as nn

from opencood.models.LevyHooks.common import (announce, check_score_mode,
                                              default_gamma, levy_pool,
                                              note_disabled, to_score)


class LevyTemporalAggregator(nn.Module):
    """
    Scores each of ``T`` past BEV maps with a 1x1 conv, turns the scores into
    Poisson intensities and pools.

    ``Lambda_time`` measures how much temporal mass a cell has accumulated -- a
    one-frame phantom scores low against a persistent object -- and the cosine
    disagreement measures frame-to-frame tracking conflict.
    """

    def __init__(self, in_channels=64, tau=0.5, score_mode='heal', gamma=None):
        super().__init__()
        self.tau = float(tau)
        assert self.tau > 0, "tau must be positive"
        self.score_mode = check_score_mode(score_mode)
        self.gamma = float(default_gamma(score_mode) if gamma is None else gamma)
        self.temporal_scorer = nn.Conv2d(in_channels, 1, kernel_size=1)

    def extra_repr(self):
        return (f"tau={self.tau:g}, gamma={self.gamma:g}, "
                f"score_mode={self.score_mode}")

    def forward(self, historical_bev_features, valid=None):
        """
        Parameters
        ----------
        historical_bev_features : torch.Tensor
            ``[B, T, C, H, W]``, oldest to newest.
        valid : torch.Tensor, optional
            ``[B, T, 1, H, W]`` bool; False where a frame does not cover a cell
            (e.g. the ego drove into the region part-way through the window).

        Returns
        -------
        fused : torch.Tensor          ``[B, C, H, W]``
        evidence : torch.Tensor       ``[B, 1, H, W]``
        disagreement : torch.Tensor   ``[B, 1, H, W]``
        """
        B, T, C, H, W = historical_bev_features.shape
        flat = historical_bev_features.reshape(B * T, C, H, W)
        raw = self.temporal_scorer(flat).reshape(B, T, 1, H, W)
        scores = to_score(raw, self.score_mode)

        fused, evidence, disagreement = levy_pool(
            historical_bev_features, scores, pool_dim=1, channel_dim=2,
            tau=self.tau, gamma=self.gamma, valid=valid,
            return_disagreement=True)
        return fused, evidence, disagreement


def build_temporal_aggregator(cfg, in_channels=64, where='(unwired)'):
    """
    Build a :class:`LevyTemporalAggregator` from a yaml sub-dict, or ``None``.

    ::

        levy_temporal:
          enable: true
          in_channels: 64
          score_mode: heal
          temp_gamma: 4.0
          rate_tau: 0.5

    Provided for symmetry with the other hooks; nothing calls it yet, since the
    dataloader carries no frame history.
    """
    cfg = cfg or {}
    if not cfg.get('enable', False):
        note_disabled('temporal', where, cfg)
        return None
    agg = LevyTemporalAggregator(
        in_channels=int(cfg.get('in_channels', in_channels)),
        tau=float(cfg.get('rate_tau', 0.5)),
        score_mode=cfg.get('score_mode') or 'heal',
        gamma=cfg.get('temp_gamma', None),
    )
    return announce('temporal', agg, where, agg.extra_repr())
