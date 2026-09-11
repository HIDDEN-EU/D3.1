"""
Hook 2 -- LevyFeatureGateway (blueprint Sec. 5).

Gates an agent's own BEV features against its own evidence, before they are
warped into the ego frame -- i.e. at the point where a real deployment would
transmit.

    call site   opencood/models/fuse_modules/levy_pyramid_fuse.py
                LevyPyramidFusion.forward_collab, inside the per-level loop
    built by    LevyPyramidFusion.__init__
    config      model.args.fusion_backbone.levy.gateway
    disable     drop the block, or set `enable: false`

Zero retraining: no parameters, so the state_dict is unchanged either way.

Two objects live here. :class:`LevyFeatureGateway` is the operator -- gate one
feature map against one score map. :class:`GatewayHook` is the *policy* around
it: which pyramid levels to touch, which agents are exempt, and the bandwidth
bookkeeping. Keeping the policy here rather than in the fusion module is what
lets the call site be three lines, and it puts every "who gets gated" decision in
one readable place.
"""

import math

import torch
import torch.nn as nn

from opencood.models.LevyHooks.common import (announce, check_score_mode,
                                              default_gamma, note_disabled,
                                              to_score)


class LevyFeatureGateway(nn.Module):
    """
    Pre-transmission gate on one agent's own BEV features::

        Lambda_j(s)   = tau * exp(gamma * S_j(s))
        F_gated(s)    = F_j(s) * (1 - exp(-Lambda_j(s)))^k
        M_j(s)        = 1[ Lambda_j(s) >= Theta ]

    ``1 - exp(-Lambda)`` is the probability that a Poisson(``Lambda``) draw is
    non-empty -- "is there anything at all here?" -- so the gate attenuates cells
    the agent has no evidence for while leaving confidently-occupied cells at
    ~1.0. Unlike the downstream operator this is a *single-agent* quantity: there
    is no sum over agents, so ``Lambda`` is a monotone function of ``S`` alone and
    the gate is exactly a fixed, parameterless nonlinearity applied to the score.

    Calibration is the whole game, and it does not survive a change of score
    range (blueprint Sec. 2). With ``tau = 0.5``::

        heal,  gamma=4.0, S in (0, 1]   Lambda in [0.5, 27.3]  gate in [0.63, 1.0]
        logit, gamma=0.25, S ~ -5       Lambda ~ 0.14          gate ~ 0.13

    The second row is the trap. A frozen checkpoint's backbone and heads were
    fitted to un-gated feature magnitudes; multiplying the *whole* map by ~0.13 is
    not a filter, it is a global rescale that the downstream BatchNorm statistics
    have never seen -- measured at AP70 0.78 against a 0.93 baseline. ``gate_ref``
    exists for that case: it divides the gate by its value at a reference score,
    so a cell at ``S = gate_ref`` passes at unity and only cells *below* the
    reference are attenuated. ``gate_ref=None`` (the default) is the literal
    blueprint formula.

    Parameters
    ----------
    gate_power : float
        ``k``. Larger ``k`` sharpens the gate; ``k = 0`` disables it.
    pruning_threshold : float
        ``Theta`` for the reported binary transmission mask. The mask is
        diagnostic (bandwidth accounting) and does not modify the features unless
        ``hard_prune`` is set. NOTE this threshold is composition-dependent: at
        ``gamma=4`` it keeps 0.9% of level-0 cells on a LiDAR-only base and 41.5%
        on a camera-bearing one. Fit it per composition.
    hard_prune : bool
        Also zero the features outside the mask, i.e. actually transmit sparsely.
        Measured at -0.10 AP70 with the inherited threshold; do not enable it
        without fitting ``pruning_threshold`` first.
    gate_ref : float or None
        Reference score at which the gate is normalised to 1, in the same units as
        ``S`` (post-sigmoid for ``'heal'``, raw logit for ``'logit'``).
    """

    def __init__(self, gate_power=0.5, pruning_threshold=0.6, score_mode='heal',
                 gamma=None, tau=0.5, hard_prune=False, gate_ref=None):
        super().__init__()
        self.k = float(gate_power)
        self.threshold = float(pruning_threshold)
        self.score_mode = check_score_mode(score_mode)
        self.gamma = float(default_gamma(score_mode) if gamma is None else gamma)
        self.tau = float(tau)
        assert self.tau > 0, "tau must be positive"
        self.hard_prune = bool(hard_prune)
        self.gate_ref = None if gate_ref is None else float(gate_ref)

        if self.gate_ref is None:
            self._gate_norm = 1.0
        else:
            lam_ref = self.tau * math.exp(self.gamma * self.gate_ref)
            self._gate_norm = (1.0 - math.exp(-lam_ref)) ** self.k

    def extra_repr(self):
        return (f"k={self.k:g}, gamma={self.gamma:g}, tau={self.tau:g}, "
                f"score_mode={self.score_mode}, threshold={self.threshold:g}, "
                f"hard_prune={self.hard_prune}, gate_ref={self.gate_ref}")

    def compute_local_evidence(self, local_foreground_map):
        """
        ``Lambda_j = tau * exp(gamma * S_j)`` from the agent's raw occupancy head
        output. ``score_mode`` decides whether the raw map is squashed first.
        """
        scores = to_score(local_foreground_map, self.score_mode)
        return self.tau * torch.exp((self.gamma * scores).clamp(max=60.0))

    def forward(self, local_bev_feature, local_foreground_map, valid=None):
        """
        Parameters
        ----------
        local_bev_feature : torch.Tensor
            ``[N, C, H, W]`` the agent's own BEV features, in its own frame.
        local_foreground_map : torch.Tensor
            ``[N, 1, H, W]`` raw occupancy-head output for the same grid.
        valid : torch.Tensor, optional
            ``[N, 1, H, W]``; cells the agent does not observe are left untouched
            (they are already zero and are masked out downstream anyway, so
            attenuating them further only wastes the gate's dynamic range).

        Returns
        -------
        gated_feature : torch.Tensor      ``[N, C, H, W]``
        transmission_mask : torch.Tensor  ``[N, 1, H, W]`` in {0, 1}
        local_evidence : torch.Tensor     ``[N, 1, H, W]``
        """
        local_evidence = self.compute_local_evidence(local_foreground_map)

        gate = (1.0 - torch.exp(-local_evidence)) ** self.k
        if self._gate_norm != 1.0:
            gate = gate / self._gate_norm
        if valid is not None:
            gate = torch.where(valid, gate, torch.ones_like(gate))

        gated_feature = local_bev_feature * gate

        transmission_mask = (local_evidence >= self.threshold).to(gate.dtype)
        if self.hard_prune:
            gated_feature = gated_feature * transmission_mask

        return gated_feature, transmission_mask, local_evidence


class GatewayHook(nn.Module):
    """
    Which agents and which pyramid levels the gate is allowed to touch.

    Measured policy, in order of how much it matters:

    ``cameras_only``
        Gate camera agents and nobody else. This is the setting to use. On a
        LiDAR-only alliance nothing is a camera, so the hook becomes an *exact*
        no-op -- verified bit-identical to fifteen decimal places -- which makes
        one configuration safe to leave on regardless of composition. On the
        m1-m4 alliance it gives +0.0132 AP70 and -3.4% FP, against +0.0062 and
        +1.0% for ``ego_exempt``. The difference is m3: ego-exemption still gates
        a SECOND LiDAR collaborator, whose Lambda sits on the tau floor and so
        carries no signal to gate on.

    ``ego_exempt``
        The ego never broadcasts to itself, so "pre-transmission" taken literally
        leaves it alone. Superseded by ``cameras_only`` but kept because it is the
        blueprint's own reading and the two differ only when a LiDAR
        *collaborator* is present.

    ``levels``
        Which pyramid levels to gate; ``None`` means all. Restricting to ``[0]``
        is what keeps the FP cost at +1.0% instead of +20.9% -- the coarse levels
        are too blunt for a per-cell evidence decision.
    """

    def __init__(self, gate, num_levels, levels=None, ego_exempt=False,
                 cameras_only=False):
        super().__init__()
        self.gate = gate
        self.levels = (set(range(num_levels)) if levels is None
                       else set(int(l) for l in levels))
        self.ego_exempt = bool(ego_exempt)
        self.cameras_only = bool(cameras_only)
        # Bandwidth accounting, refreshed per forward by begin_forward().
        self.stats = []

    def extra_repr(self):
        return (f"{self.gate.extra_repr()}, levels={sorted(self.levels)}, "
                f"ego_exempt={self.ego_exempt}, cameras_only={self.cameras_only}")

    def begin_forward(self):
        """Clear the per-forward bandwidth stats. Call once before the level loop."""
        self.stats = []

    def _exempt_mask(self, n_agents, device, record_len, agent_modality_list,
                     cam_crop_info):
        """Bool over agents: True where the features pass through ungated."""
        keep = torch.zeros(n_agents, dtype=torch.bool, device=device)
        if self.ego_exempt:
            # Agent 0 of each record_len group is the ego.
            offset = 0
            for n in record_len:
                keep[offset] = True
                offset += int(n)
        if self.cameras_only:
            # cam_crop_info is keyed by the camera modalities, so it is the
            # authority on which agents carry a camera. It is empty for a
            # LiDAR-only model, which makes every agent exempt and the whole hook
            # an exact no-op.
            cam_set = set(cam_crop_info.keys()) if cam_crop_info else set()
            for j, modality in enumerate(agent_modality_list or []):
                if modality not in cam_set:
                    keep[j] = True
        return keep

    def apply(self, level, features, occ_map, record_len,
              agent_modality_list=None, cam_crop_info=None):
        """
        Gate one pyramid level's features in place of the caller's tensor.

        Returns ``features`` unchanged when this level is not gated, so the call
        site needs no branch of its own.

        ``occ_map`` is the raw occupancy logits the fusion already computed for
        this level. The fusion's own ``score`` is deliberately left ungated: the
        gate attenuates the feature a background cell *contributes*, it does not
        re-weight which agent wins the cell, so the downstream fusion weights
        ``p_j`` stay exactly what they would have been.
        """
        if level not in self.levels:
            return features

        gated, tx_mask, local_evidence = self.gate(features, occ_map)

        keep = self._exempt_mask(gated.shape[0], gated.device, record_len,
                                 agent_modality_list, cam_crop_info)
        if keep.any():
            gated = torch.where(keep.view(-1, 1, 1, 1), features, gated)

        self.stats.append({
            'level': level,
            'kept_frac': float(tx_mask.mean().detach()),
            'evidence_mean': float(local_evidence.mean().detach()),
            'evidence_max': float(local_evidence.max().detach()),
        })
        return gated


def build_gateway_hook(cfg, num_levels, default_score_mode='heal',
                       where='LevyPyramidFusion'):
    """
    Build a :class:`GatewayHook` from the ``levy.gateway`` yaml block, or ``None``.

    ::

        gateway:
          enable: false
          gate_power: 0.5          # k
          temp_gamma: 4.0          # defaults per score_mode
          rate_tau: 0.5
          score_mode: null         # null = inherit the fusion's own score_mode
          pruning_threshold: 0.6
          hard_prune: false
          gate_ref: null
          levels: [0]              # null = every level
          ego_exempt: false
          cameras_only: true       # recommended; no-op without camera agents
    """
    cfg = cfg or {}
    if not cfg.get('enable', False):
        note_disabled('gateway', where, cfg)
        return None

    # `or`, not a get() default: the documented yaml block spells the "use the
    # fusion's own mode" case as an explicit `score_mode: null`, which get()
    # would hand back as None.
    score_mode = cfg.get('score_mode') or default_score_mode
    gate = LevyFeatureGateway(
        gate_power=float(cfg.get('gate_power', 0.5)),
        pruning_threshold=float(cfg.get('pruning_threshold', 0.6)),
        score_mode=score_mode,
        gamma=cfg.get('temp_gamma', None),
        tau=float(cfg.get('rate_tau', 0.5)),
        hard_prune=bool(cfg.get('hard_prune', False)),
        gate_ref=cfg.get('gate_ref', None),
    )
    hook = GatewayHook(
        gate,
        num_levels=num_levels,
        levels=cfg.get('levels', None),
        ego_exempt=cfg.get('ego_exempt', False),
        cameras_only=cfg.get('cameras_only', False),
    )
    return announce('gateway', hook, where, hook.extra_repr())
