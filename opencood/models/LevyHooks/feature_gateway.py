"""
Gates an agent's own BEV features against its own evidence, before they are
warped into the ego frame -- i.e. at the point where a real deployment would
transmit.

    call site   opencood/models/fuse_modules/levy_pyramid_fuse.py
                LevyPyramidFusion.forward_collab, inside the per-level loop
    built by    LevyPyramidFusion.__init__
    config      model.args.fusion_backbone.levy.gateway
    disable     drop the block, or set `enable: false`

Zero retraining: no parameters, so the state_dict is unchanged either way.

Two kinds of object live here. :class:`LevyFeatureGateway` and
:class:`SigmoidFeatureGateway` are interchangeable operators (``gate_fn``) -- gate
one feature map against one score map. :class:`GatewayHook` is the *policy* around
it: which pyramid levels to touch, which agents are exempt, and the bandwidth
bookkeeping. Keeping the policy here rather than in the fusion module is what
lets the call site be three lines, and it puts every "who gets gated" decision in
one readable place.
"""

import math

import torch
import torch.nn as nn

from opencood.models.LevyHooks.common import announce, note_disabled


class LevyFeatureGateway(nn.Module):
    """
    Pre-transmission gate on one agent's own BEV features::

        S_j(s)        = sigmoid(occ_j(s))
        Lambda_j(s)   = tau * exp(gamma * S_j(s))
        F_gated(s)    = F_j(s) * (1 - exp(-Lambda_j(s)))^k
        M_j(s)        = 1[ Lambda_j(s) >= Theta ]

    ``1 - exp(-Lambda)`` is the probability that a Poisson(``Lambda``) draw is
    non-empty -- "is there anything at all here?" -- so the gate attenuates cells
    the agent has no evidence for while leaving confidently-occupied cells at
    ~1.0. Unlike the downstream operator this is a *single-agent* quantity: there
    is no sum over agents, so ``Lambda`` is a monotone function of ``S`` alone and
    the gate is exactly a fixed, parameterless nonlinearity applied to the score.

    With ``tau = 0.5, gamma = 4``, ``S in (0, 1]`` gives ``Lambda in [0.5, 27.3]``
    and a gate in ``[0.63, 1.0]``. A frozen checkpoint's backbone and heads were
    fitted to un-gated feature magnitudes, so the floor matters: a gate that
    scales the *whole* map down is a global rescale the downstream BatchNorm
    statistics have never seen, not a filter. ``gate_ref`` divides the gate by its
    value at a reference score, so a cell at ``S = gate_ref`` passes at unity and
    only cells *below* the reference are attenuated. ``gate_ref=None`` (the
    default) is the literal blueprint formula.

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
        Reference score ``S`` (post-sigmoid) at which the gate is normalised to 1.
    """

    def __init__(self, gate_power=0.5, pruning_threshold=0.6, gamma=4.0, tau=0.5,
                 hard_prune=False, gate_ref=None):
        super().__init__()
        self.k = float(gate_power)
        self.threshold = float(pruning_threshold)
        self.gamma = float(gamma)
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
                f"threshold={self.threshold:g}, "
                f"hard_prune={self.hard_prune}, gate_ref={self.gate_ref}")

    def compute_local_evidence(self, local_foreground_map):
        """
        ``Lambda_j = tau * exp(gamma * sigmoid(occ_j))`` from the agent's raw
        occupancy head output.
        """
        return self.tau * torch.exp(self.gamma * torch.sigmoid(local_foreground_map))

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


class SigmoidFeatureGateway(nn.Module):
    """
    The same pre-transmission gate without the Levy evidence::

        g(s)   = g_min + (1 - g_min) * sigmoid((x_j(s) - t) / T)
        F(s)   = F_j(s) * g(s)
        M_j(s) = 1[ x_j(s) >= m ]

    with ``x_j`` the raw occupancy logit. With one agent and one cell there is no
    sum for ``Lambda`` to carry, so the Levy gate above is only a particular
    monotone curve of ``x``; this one sets that curve's three degrees of freedom
    directly, in logit units that can be read off a score histogram:

    ``g_min`` the gate on cells the agent is sure are empty (the tau floor), i.e.
    how far the background is pushed from the magnitudes the frozen BatchNorm
    was fitted to; ``t`` the logit where the gate is halfway up; ``T`` the
    width of the transition. ``M_j`` is a Where2comm-style mask on the same logit.

    Parameters
    ----------
    gate_floor : float
        ``g_min`` in [0, 1]. ``1.0`` disables the gate.
    gate_center : float
        ``t``, raw logit. The default -3.0 matches the Levy mask's cut-off at its
        defaults (``sigmoid(x) = ln(Theta/tau)/gamma ~ 0.046``).
    gate_temp : float
        ``T`` > 0, in logits.
    mask_logit : float or None
        ``m``; ``None`` uses ``gate_center``. Diagnostic unless ``hard_prune``.
    hard_prune : bool
        Also zero the features outside the mask.
    """

    def __init__(self, gate_floor=0.5, gate_center=-3.0, gate_temp=1.0,
                 mask_logit=None, hard_prune=False):
        super().__init__()
        self.g_min = float(gate_floor)
        assert 0.0 <= self.g_min <= 1.0, "gate_floor must lie in [0, 1]"
        self.t = float(gate_center)
        self.T = float(gate_temp)
        assert self.T > 0, "gate_temp must be positive"
        self.mask_logit = self.t if mask_logit is None else float(mask_logit)
        self.hard_prune = bool(hard_prune)

    def extra_repr(self):
        return (f"gate_fn=sigmoid, g_min={self.g_min:g}, t={self.t:g}, "
                f"T={self.T:g}, mask_logit={self.mask_logit:g}, "
                f"hard_prune={self.hard_prune}")

    def forward(self, local_bev_feature, local_foreground_map, valid=None):
        """
        Same contract as :meth:`LevyFeatureGateway.forward`, except the third
        output is the occupancy probability ``sigmoid(x_j)`` rather than
        ``Lambda_j``.
        """
        x = local_foreground_map
        gate = self.g_min + (1.0 - self.g_min) * torch.sigmoid((x - self.t) / self.T)
        if valid is not None:
            gate = torch.where(valid, gate, torch.ones_like(gate))

        gated_feature = local_bev_feature * gate

        transmission_mask = (x >= self.mask_logit).to(gate.dtype)
        if self.hard_prune:
            gated_feature = gated_feature * transmission_mask

        return gated_feature, transmission_mask, torch.sigmoid(x)


class GatewayHook(nn.Module):
    """
    Which agents and which pyramid levels the gate is allowed to touch.

    Measured policy, in order of how much it matters:

    ``cameras_only``
        Gate camera agents and nobody else. On a
        LiDAR-only alliance nothing is a camera, so the hook becomes an *exact*
        no-op. 
        On the m1-m4 alliance it gives +0.0132 AP70 and -3.4% FP, against +0.0062 and
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

        gated, tx_mask, signal = self.gate(features, occ_map)

        keep = self._exempt_mask(gated.shape[0], gated.device, record_len,
                                 agent_modality_list, cam_crop_info)
        if keep.any():
            gated = torch.where(keep.view(-1, 1, 1, 1), features, gated)

        self.stats.append({
            'level': level,
            'kept_frac': float(tx_mask.mean().detach()),
            # Lambda for the Levy gate, sigmoid(occ) for the sigmoid gate.
            'signal_mean': float(signal.mean().detach()),
            'signal_max': float(signal.max().detach()),
        })
        return gated


def build_gateway_hook(cfg, num_levels, where='LevyPyramidFusion'):
    """
    Build a :class:`GatewayHook` from the ``levy.gateway`` yaml block, or ``None``.
        gateway:
          enable: false
          gate_fn: levy            # levy | sigmoid
          # -- gate_fn: levy --
          gate_power: 0.5          # k
          temp_gamma: 4.0
          rate_tau: 0.5
          pruning_threshold: 0.6
          hard_prune: false
          gate_ref: null
          # -- gate_fn: sigmoid (raw occupancy logits) --
          gate_floor: 0.5          # g_min
          gate_center: -3.0        # t
          gate_temp: 1.0           # T
          mask_logit: null         # null = gate_center
          # hard_prune applies to both
          # -- policy --
          levels: [0]              # null = every level
          ego_exempt: false
          cameras_only: true       # recommended; no-op without camera agents
    """
    cfg = cfg or {}
    if not cfg.get('enable', False):
        note_disabled('gateway', where, cfg)
        return None

    gate_fn = cfg.get('gate_fn') or 'levy'
    if gate_fn == 'levy':
        gate = LevyFeatureGateway(
            gate_power=float(cfg.get('gate_power', 0.5)),
            pruning_threshold=float(cfg.get('pruning_threshold', 0.6)),
            gamma=float(cfg.get('temp_gamma', 4.0)),
            tau=float(cfg.get('rate_tau', 0.5)),
            hard_prune=bool(cfg.get('hard_prune', False)),
            gate_ref=cfg.get('gate_ref', None),
        )
    elif gate_fn == 'sigmoid':
        gate = SigmoidFeatureGateway(
            gate_floor=float(cfg.get('gate_floor', 0.5)),
            gate_center=float(cfg.get('gate_center', -3.0)),
            gate_temp=float(cfg.get('gate_temp', 1.0)),
            mask_logit=cfg.get('mask_logit', None),
            hard_prune=bool(cfg.get('hard_prune', False)),
        )
    else:
        raise ValueError(f"levy.gateway.gate_fn must be 'levy' or 'sigmoid', "
                         f"got {gate_fn!r}")
    hook = GatewayHook(
        gate,
        num_levels=num_levels,
        levels=cfg.get('levels', None),
        ego_exempt=cfg.get('ego_exempt', False),
        cameras_only=cfg.get('cameras_only', False),
    )
    return announce('gateway', hook, where, hook.extra_repr())
