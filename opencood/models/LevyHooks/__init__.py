"""
Upstream (single-agent, pre-transmission) Levy attention hooks for HEAL.

Companion to :mod:`opencood.models.fuse_modules.levy_pyramid_fuse`, which applies
the same Poisson-intensity construction *downstream*, across agents, after the
features have been warped into the ego frame. The hooks here apply it *upstream*,
inside one agent, before anything is transmitted.

Reference: ``levy-heal-fusion-blueprint.md`` Sections 3-6.

One file per hook
-----------------
=================== ======================= ===================================
file                config key              call site
=================== ======================= ===================================
depth_pool.py       encoder_args            sub_modules/lss_submodule.py
                    .levy_depth             CamEncode*.get_depth_dist
feature_gateway.py  fusion_backbone         fuse_modules/levy_pyramid_fuse.py
                    .levy.gateway           LevyPyramidFusion.forward_collab
intra_modality.py   <modality>.levy_intra   fused_pyramid_collab.py
                                            FusedPyramidCollab.encode_modality
temporal_aggregator not wired -- the OPV2V pipeline is single-frame
.py
=================== ======================= ===================================

``common.py`` holds the pooling primitive they all share, plus the
enable/disable reporting below. No hook logic lives there.

Is a hook on?
-------------
Every hook is built by a ``build_*`` function that returns ``None`` when its
config block is absent or carries ``enable: false``. Host modules therefore hold
``None`` or an object, never a flag, and disabling a hook is a one-line config
change that removes it from the graph entirely rather than short-circuiting it.

Each builder prints one greppable line as it constructs, so the question is
answered from the log rather than by reading model code::

    [LevyHook] gateway        ENABLED   LevyPyramidFusion   k=0.5, gamma=4, ...
    [LevyHook] depth_pool     disabled  LiftSplatShoot/Resnet101   (block present, enable: false)

and in-process::

    from opencood.models import LevyHooks
    print(LevyHooks.summary())

A block that is simply absent prints nothing -- only a block that is present and
switched off is worth a line, since that is the case where someone expects the
hook to be running.

What is measured
----------------
See the dossier for the full picture. The two findings that change how these are
configured:

* **Use ``cameras_only`` on the gateway.** It gates camera agents and nobody
  else, which makes it an exact no-op on a LiDAR-only alliance (verified
  bit-identical to 15 decimal places) and beats ``ego_exempt`` where cameras are
  present: +0.0132 AP70 / -3.4% FP against +0.0062 / +1.0%.
* **gamma, the pruning threshold and the downstream beta are properties of the
  alliance composition, not of the operator.** At gamma=4 the same threshold
  keeps 0.9% of cells on a LiDAR-only base and 41.5% on a camera-bearing one.
  Re-measure before carrying any of them across.

Self-tests: ``python -m opencood.models.LevyHooks.selftest``.

License: TDG-Attribution-NonCommercial-NoDistrib
"""

from opencood.models.LevyHooks.common import (LOG_ZERO, SCORE_MODES,
                                              active_hooks, announce,
                                              check_score_mode, default_gamma,
                                              levy_pool, log_intensity,
                                              note_disabled, reset_registry,
                                              summary, to_score)
from opencood.models.LevyHooks.depth_pool import LevyDepthPool, build_depth_pool
from opencood.models.LevyHooks.feature_gateway import (GatewayHook,
                                                       LevyFeatureGateway,
                                                       build_gateway_hook)
from opencood.models.LevyHooks.intra_modality import (
    LevyIntraModalityFusion, build_intra_modality_fusion)
from opencood.models.LevyHooks.temporal_aggregator import (
    LevyTemporalAggregator, build_temporal_aggregator)

__all__ = [
    # hooks
    'LevyDepthPool', 'LevyFeatureGateway', 'GatewayHook',
    'LevyIntraModalityFusion', 'LevyTemporalAggregator',
    # builders -- each returns None when its config block disables it
    'build_depth_pool', 'build_gateway_hook', 'build_intra_modality_fusion',
    'build_temporal_aggregator',
    # shared primitives
    'levy_pool', 'to_score', 'log_intensity', 'default_gamma',
    'check_score_mode', 'LOG_ZERO', 'SCORE_MODES',
    # enable/disable reporting
    'announce', 'note_disabled', 'active_hooks', 'summary', 'reset_registry',
]
