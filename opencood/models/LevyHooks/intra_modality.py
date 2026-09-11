"""
Hook 4 -- LevyIntraModalityFusion (blueprint Sec. 6).

Levy fusion across one agent's *own* sensors, replacing the concat + 1x1 conv
that merges its LiDAR and camera BEV maps.

    call site   opencood/models/fused_pyramid_collab.py
                FusedPyramidCollab.encode_modality
    built by    FusedPyramidCollab.__init__
    config      model.args.<modality>.levy_intra
    disable     drop the block, or set `enable: false` -- the concat + conv path
                is restored exactly

**Not zero-retraining.** The ``scorers`` are fresh 1x1 convs and this bypasses
the trained ``fusion_conv`` entirely, so a checkpoint fitted with concat-fusion
has to be retrained before the numbers mean anything. Evaluating it on frozen
weights measures destroyed weights, not the operator. It has never been
benchmarked here for that reason; it is verified to construct, register as
trainable, and produce the right shapes.
"""

import torch
import torch.nn as nn

from opencood.models.LevyHooks.common import (announce, check_score_mode,
                                              default_gamma, levy_pool,
                                              note_disabled, to_score)


class LevyIntraModalityFusion(nn.Module):
    """
    Structurally identical to the downstream :class:`LevyPyramidFusion` step, but
    the sum runs over modalities inside a single vehicle rather than over
    vehicles, so ``Lambda_intra`` reads as "how many of *my* sensors see this
    cell" and the cosine disagreement as LiDAR/camera conflict -- typically
    camera depth smear landing where the point cloud is empty.

    ``scorers`` are fresh 1x1 convs unless per-modality foreground maps are
    supplied to :meth:`forward`; supplying existing occupancy heads' outputs as
    ``scores`` makes it parameter-free.
    """

    def __init__(self, in_channels=64, num_modalities=2, tau=0.5,
                 score_mode='heal', gamma=None, learn_scorers=True):
        super().__init__()
        self.tau = float(tau)
        assert self.tau > 0, "tau must be positive"
        self.score_mode = check_score_mode(score_mode)
        self.gamma = float(default_gamma(score_mode) if gamma is None else gamma)
        self.num_modalities = int(num_modalities)
        if learn_scorers:
            self.scorers = nn.ModuleList([
                nn.Conv2d(in_channels, 1, kernel_size=1)
                for _ in range(self.num_modalities)])
        else:
            self.scorers = None

    def extra_repr(self):
        return (f"tau={self.tau:g}, gamma={self.gamma:g}, "
                f"score_mode={self.score_mode}, M={self.num_modalities}")

    def forward(self, modality_features, scores=None, valid=None):
        """
        Parameters
        ----------
        modality_features : torch.Tensor or list of torch.Tensor
            ``[N, M, C, H, W]``, or a list of ``M`` tensors ``[N, C, H, W]``.
        scores : torch.Tensor, optional
            ``[N, M, 1, H, W]`` raw per-modality foreground maps. When omitted the
            learned ``scorers`` produce them.
        valid : torch.Tensor, optional
            ``[N, M, 1, H, W]`` bool; a modality that does not cover a cell (camera
            crop, LiDAR blind spot) carries zero intensity there.

        Returns
        -------
        fused : torch.Tensor          ``[N, C, H, W]``
        evidence : torch.Tensor       ``[N, 1, H, W]``
        disagreement : torch.Tensor   ``[N, 1, H, W]``
        """
        if isinstance(modality_features, (list, tuple)):
            modality_features = torch.stack(modality_features, dim=1)
        N, M, C, H, W = modality_features.shape
        assert M == self.num_modalities, \
            f"expected {self.num_modalities} modalities, got {M}"

        if scores is None:
            assert self.scorers is not None, \
                "LevyIntraModalityFusion built with learn_scorers=False needs " \
                "explicit per-modality scores"
            raw = torch.stack([self.scorers[m](modality_features[:, m])
                               for m in range(M)], dim=1)  # [N, M, 1, H, W]
        else:
            raw = scores
        s = to_score(raw, self.score_mode)  # [N, M, 1, H, W]

        fused, evidence, disagreement = levy_pool(
            modality_features, s, pool_dim=1, channel_dim=2,
            tau=self.tau, gamma=self.gamma, valid=valid,
            return_disagreement=True)
        return fused, evidence, disagreement


def build_intra_modality_fusion(cfg, fusion_args, where='FusedPyramidCollab'):
    """
    Build a :class:`LevyIntraModalityFusion` from the ``levy_intra`` block, or
    ``None``.

    ::

        levy_intra:
          enable: true
          score_mode: heal
          temp_gamma: 4.0
          rate_tau: 0.5

    ``fusion_args`` is the modality's existing concat-fusion config; it supplies
    the channel count and is checked for the symmetry this operator needs --
    concat-fusion tolerates two modalities of different widths, Levy pooling does
    not, since it averages them in a shared space.
    """
    cfg = cfg or {}
    if not cfg.get('enable', False):
        note_disabled('intra_modality', where, cfg)
        return None

    assert fusion_args['in_channels'] == 2 * fusion_args['out_channels'], \
        ("levy_intra needs the two modalities to share a channel count: expected "
         f"in_channels == 2 * out_channels, got {fusion_args['in_channels']} and "
         f"{fusion_args['out_channels']}")

    fusion = LevyIntraModalityFusion(
        in_channels=fusion_args['out_channels'],
        num_modalities=2,
        tau=float(cfg.get('rate_tau', 0.5)),
        score_mode=cfg.get('score_mode') or 'heal',
        gamma=cfg.get('temp_gamma', None),
    )
    return announce('intra_modality', fusion, where,
                    fusion.extra_repr() + ' (replaces fusion_conv; NEEDS TRAINING)')
