"""
Shared machinery for every Levy hook: the pooling primitive and the
enable/disable reporting every hook uses.

Nothing in here is a hook. The hooks live one-per-file alongside it
(``depth_pool.py``, ``feature_gateway.py``, ``temporal_aggregator.py``,
``intra_modality.py``) and all of them go through :func:`levy_pool`.

The shared construction
-----------------------
Every operator exponentiates a score ``S`` into an unnormalised Poisson
intensity, sums it over whatever axis it is pooling, and reads three things off
the result::

    lambda_k = tau * exp(gamma * S_k)                  per-atom intensity
    Lambda   = sum_k lambda_k                          evidence (unnormalised mass)
    p_k      = lambda_k / Lambda                       pooling weights (normalised)
    F_bar    = sum_k p_k F_k                           pooled feature (mean path)
    tr Sigma = 1 - || sum_k p_k F_k/||F_k|| ||^2       cosine disagreement in [0, 1]

``Lambda`` is what row-normalised softmax throws away: it is the absolute mass of
evidence, and therefore the only channel that can tell "nothing here" from
"something here", since ``p_k`` is invariant to adding a constant to every
``S_k``.

Everything is held in log space (``log lambda = log tau + gamma * S``, evidence
via ``logsumexp``) so that a large ``gamma`` on unbounded logits cannot overflow
-- the blueprint's direct ``tau * torch.exp(gamma * scores)`` overflows fp16 at
``gamma * S > 11`` and fp32 at ``> 88``, which ``gamma=4`` reaches on any logit
above 22.

Temperature calibration (blueprint Sec. 2)
------------------------------------------
``gamma`` has to be matched to the dynamic range of ``S`` so that
``gamma * range(S)`` lands around 2-4 nats::

    score_mode='heal'    S = sigmoid(occ) in (0, 1]   range ~1    -> gamma ~2-4
    score_mode='logit'   S = raw occ logits           range ~10   -> gamma ~0.25

``score_mode`` is explicit rather than sniffed from ``S.max() <= 1.0`` as in the
blueprint listing: on a batch where every logit happens to be negative that test
silently misfires, and the resulting gate is wrong by orders of magnitude with no
error.

Measured on OPV2V-H, ``gamma`` and the pruning threshold are properties of the
*alliance composition*, not of the operator -- at ``gamma=4`` the same threshold
keeps 0.9% of cells on a LiDAR-only base and 41.5% on a camera-bearing one. Do
not carry either constant between compositions without re-measuring.

License: TDG-Attribution-NonCommercial-NoDistrib
"""

import math

import torch
import torch.nn.functional as F

# Additive stand-in for log(0): finite (so fp16 stays clean) yet exp()es to 0.
LOG_ZERO = -1e4

SCORE_MODES = ('heal', 'logit')


# --------------------------------------------------------------------------- #
# Enable / disable reporting
#
# Every hook is built by a `build_*` function that returns None when its config
# block is absent or `enable: false`. Those builders call `announce` on the way
# out, so a run prints one greppable line per hook and the question "is this
# hook on?" is answered by `grep '\[LevyHook\]'` over the log rather than by
# reading the model code.
# --------------------------------------------------------------------------- #

_ACTIVE = []


def announce(hook_name, instance, where, detail=''):
    """
    Record and print that ``hook_name`` is live. Pass-through for ``instance``.

    Returns ``instance`` so a builder can end with ``return announce(...)``.
    ``None`` in means ``None`` out and nothing printed, which is the disabled
    path.
    """
    if instance is None:
        return None
    _ACTIVE.append({'hook': hook_name, 'where': where, 'detail': detail})
    print(f"[LevyHook] {hook_name:<14} ENABLED   {where}"
          + (f"   {detail}" if detail else ""))
    return instance


def note_disabled(hook_name, where, cfg):
    """
    Print a line when a config block exists but is switched off.

    A block that is present and disabled is worth one line -- it is the case
    where someone expects the hook to be running. A block that is absent
    entirely says nothing.
    """
    if cfg:
        print(f"[LevyHook] {hook_name:<14} disabled  {where}   "
              f"(block present, enable: false)")


def active_hooks():
    """Every hook announced in this process, in construction order."""
    return list(_ACTIVE)


def summary():
    """One-line-per-hook description of what is live. Empty string if nothing."""
    if not _ACTIVE:
        return "[LevyHook] no hooks enabled"
    return '\n'.join(
        f"[LevyHook] {h['hook']:<14} {h['where']}   {h['detail']}".rstrip()
        for h in _ACTIVE)


def reset_registry():
    """Forget what has been announced. For tests that build several models."""
    _ACTIVE.clear()


# --------------------------------------------------------------------------- #
# Intensity helpers
# --------------------------------------------------------------------------- #

def default_gamma(score_mode):
    """Blueprint Sec. 2 defaults: 4.0 for bounded sigmoid scores, 0.25 for logits."""
    return 4.0 if score_mode == 'heal' else 0.25


def check_score_mode(score_mode):
    assert score_mode in SCORE_MODES, \
        f"unknown score_mode {score_mode!r}, expected one of {SCORE_MODES}"
    return score_mode


def to_score(raw, score_mode):
    """
    Map a head's raw output to the quantity the intensity exponentiates.

    ``'heal'`` squashes to the bounded sigmoid score HEAL itself fuses on;
    ``'logit'`` passes the unnormalised logit straight through.
    """
    return torch.sigmoid(raw) if score_mode == 'heal' else raw


def log_intensity(scores, tau, gamma):
    """``log lambda = log(tau) + gamma * S``, elementwise."""
    return math.log(tau) + gamma * scores


def levy_pool(features, scores, pool_dim, channel_dim, tau=0.5, gamma=1.0,
              valid=None, eps=1e-6, return_disagreement=True):
    """
    The shared Levy pooling step, over an arbitrary axis.

    Both axes are named explicitly rather than inferred, because the two layouts
    in this codebase disagree about their order: HEAL's LSS carries
    ``[B*N, C, D, fH, fW]`` (channel *before* the pooled axis) while the BEV
    stacks carry ``[B, T, C, H, W]`` (channel *after* it).

    Parameters
    ----------
    features : torch.Tensor
        Features to pool, any rank.
    scores : torch.Tensor
        ``S``, already mapped through :func:`to_score`. Same rank as ``features``,
        size 1 on ``channel_dim``, matching ``features`` on ``pool_dim``.
    pool_dim : int
        Axis to pool over (depth bins / time steps / modalities). Non-negative.
    channel_dim : int
        Feature-channel axis, the one the cosine normalisation runs along.
        Non-negative.
    valid : torch.Tensor, optional
        Bool mask shaped like ``scores``. Invalid entries carry zero intensity, so
        they contribute neither weight nor evidence -- as opposed to being given a
        small-but-nonzero weight, which is what masking with a large negative
        score would do.
    return_disagreement : bool
        Also compute the unit-normalised cosine disagreement. Costs one extra
        normalisation over ``features``; skip it when only the pooled feature and
        the evidence are wanted.

    Returns
    -------
    pooled : torch.Tensor
        ``sum_k p_k F_k``, with ``pool_dim`` removed.
    evidence : torch.Tensor
        ``Lambda``, with ``pool_dim`` removed and ``channel_dim`` of size 1.
    disagreement : torch.Tensor or None
        ``1 - ||sum_k p_k F_hat_k||^2`` in ``[0, 1]``, same shape as ``evidence``.
        ``None`` when ``return_disagreement`` is False.
    """
    assert pool_dim >= 0 and channel_dim >= 0, "axes must be non-negative"
    assert pool_dim != channel_dim, "pool and channel axes must differ"

    log_lam = log_intensity(scores, tau, gamma)
    if valid is not None:
        log_lam = log_lam.masked_fill(~valid, LOG_ZERO)

    log_evidence = torch.logsumexp(log_lam, dim=pool_dim, keepdim=True)
    weights = torch.exp(log_lam - log_evidence)
    if valid is not None:
        weights = weights * valid.to(weights.dtype)

    pooled = torch.sum(features * weights, dim=pool_dim)
    evidence = torch.exp(log_evidence.clamp(max=60.0)).squeeze(pool_dim)

    disagreement = None
    if return_disagreement:
        # Unit-normalise first: a background cell whose feature is merely *short*
        # must not read as agreement. Every summand then has ||F_hat|| = 1, so
        # the weighted variance collapses to 1 - ||mean direction||^2 in [0, 1].
        feat_hat = F.normalize(features, p=2, dim=channel_dim, eps=eps)
        mean_dir = torch.sum(feat_hat * weights, dim=pool_dim, keepdim=True)
        disagreement = (1.0 - torch.sum(mean_dir ** 2, dim=channel_dim,
                                        keepdim=True)).clamp(0.0, 1.0)
        disagreement = disagreement.squeeze(pool_dim)

    return pooled, evidence, disagreement
