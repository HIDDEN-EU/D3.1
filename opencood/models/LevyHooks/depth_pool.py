"""
Hook 1 -- LevyDepthPool (blueprint Sec. 3).

Replaces the depth softmax inside HEAL's LSS camera encoder with normalised
Poisson intensity over the depth bins.

    call site   opencood/models/sub_modules/lss_submodule.py
                CamEncode.get_depth_dist and CamEncode_Resnet101.get_depth_dist
    built by    opencood/models/heter_encoders.py  LiftSplatShoot.__init__
    config      model.args.<modality>.encoder_args.levy_depth
    disable     drop the block, or set `enable: false`

Zero retraining: the module holds no parameters, so attaching one leaves the
state_dict byte-identical and pretrained checkpoints load unchanged.
"""

import torch

from opencood.models.LevyHooks.common import (announce, check_score_mode,
                                              default_gamma, levy_pool,
                                              log_intensity, note_disabled,
                                              to_score)
import torch.nn as nn


class LevyDepthPool(nn.Module):
    """
    Poisson-intensity pooling over camera depth bins.

    Standard LSS forces the depth weights to sum to 1 even when the depth head is
    completely uncertain, so a uniform "I have no idea" spread and a confident
    single-bin spike splat the same total mass into the BEV grid. Keeping
    ``Lambda_ray`` unnormalised alongside the weights recovers that distinction.

    Two things are worth stating plainly about what this does on *this* codebase:

    1. With ``score_mode='logit'`` the normalised weights are algebraically
       ``softmax(gamma * depth_logit)`` -- the operator degenerates to a
       temperature on the existing depth softmax, and ``gamma = 1`` reproduces
       stock LSS exactly. The new content is ``Lambda_ray``, not the weights.
       That parity is the intended control, and it holds to four decimals over
       the full 2170-frame split.
    2. HEAL's LSS builds its per-bin features as ``depth_k * x_img``, i.e. every
       depth bin along a ray shares one direction in feature space. The ray cosine
       disagreement of Sec. 3.2 is therefore identically 0 here, and is only
       meaningful for an LSS variant with per-bin context. ``forward`` returns it
       anyway (as zeros) so the interface matches the blueprint.

    Measured: flatter beats sharper. ``score_mode='logit', gamma=0.5`` gives
    -8.1% fp_rate for -0.0072 AP70 on the m1-m4 alliance; the blueprint's own
    ``score_mode='heal', gamma=4.0`` is worse at -6.6% for -0.0212.
    """

    def __init__(self, tau=0.5, score_mode='heal', gamma=None,
                 return_disagreement=False):
        super().__init__()
        self.tau = float(tau)
        assert self.tau > 0, "tau must be positive"
        self.score_mode = check_score_mode(score_mode)
        self.gamma = float(default_gamma(score_mode) if gamma is None else gamma)
        self.return_disagreement = bool(return_disagreement)

    def extra_repr(self):
        return (f"tau={self.tau:g}, gamma={self.gamma:g}, "
                f"score_mode={self.score_mode}")

    def forward(self, depth_logits, depth_features=None, depth_dim=1):
        """
        Parameters
        ----------
        depth_logits : torch.Tensor
            Raw depth-head output, ``[B*N, D, fH, fW]`` in HEAL's LSS layout
            (any layout works, name the bin axis with ``depth_dim``).
        depth_features : torch.Tensor, optional
            Per-bin context features with a channel axis at ``depth_dim``, i.e.
            ``[B*N, C, D, fH, fW]``. When omitted -- the HEAL case, where context
            is depth-independent -- only the weights and the evidence are
            computed and both returned feature/disagreement slots are ``None``.
        depth_dim : int
            Axis of the depth bins in ``depth_logits``.

        Returns
        -------
        weights : torch.Tensor
            ``p_k``, same shape as ``depth_logits``; drop-in for the softmax.
        ray_evidence : torch.Tensor
            ``Lambda_ray``, depth axis reduced with ``keepdim=True``.
        pooled_features : torch.Tensor or None
        ray_disagreement : torch.Tensor or None
        """
        scores = to_score(depth_logits, self.score_mode)
        log_lam = log_intensity(scores, self.tau, self.gamma)
        log_evidence = torch.logsumexp(log_lam, dim=depth_dim, keepdim=True)
        weights = torch.exp(log_lam - log_evidence)
        ray_evidence = torch.exp(log_evidence.clamp(max=60.0))

        if depth_features is None:
            return weights, ray_evidence, None, None

        # LSS layout: features are [B*N, C, D, fH, fW], i.e. the channel axis sits
        # one *before* the depth axis of the logits.
        assert depth_features.dim() == depth_logits.dim() + 1, \
            "depth_features must carry a channel axis in front of the depth axis"
        channel_dim = depth_dim
        pool_dim = depth_dim + 1
        pooled, _, disagreement = levy_pool(
            depth_features, scores.unsqueeze(channel_dim),
            pool_dim=pool_dim, channel_dim=channel_dim,
            tau=self.tau, gamma=self.gamma,
            return_disagreement=self.return_disagreement)
        return weights, ray_evidence, pooled, disagreement


def build_depth_pool(cfg, where='LiftSplatShoot'):
    """
    Build a :class:`LevyDepthPool` from an encoder-args sub-dict, or ``None``.

    Accepts the ``levy_depth`` block on an LSS encoder's ``encoder_args``::

        levy_depth:
          enable: true
          score_mode: logit        # 'logit' == a temperature on the LSS softmax
          temp_gamma: 1.0          # 1.0 in logit mode reproduces stock LSS exactly
          rate_tau: 0.5

    The defaults are deliberately the identity, so switching ``enable`` on with
    nothing else set is a parity check rather than a change.
    """
    cfg = cfg or {}
    if not cfg.get('enable', False):
        note_disabled('depth_pool', where, cfg)
        return None
    score_mode = cfg.get('score_mode') or 'logit'
    gamma = cfg.get('temp_gamma') or (1.0 if score_mode == 'logit' else None)
    pool = LevyDepthPool(
        tau=float(cfg.get('rate_tau', 0.5)),
        score_mode=score_mode,
        gamma=gamma,
        return_disagreement=False,
    )
    return announce('depth_pool', pool, where, pool.extra_repr())
