"""
Upstream (single-agent, pre-transmission) Levy hooks for HEAL.

Companion to :mod:`opencood.models.fuse_modules.levy_pyramid_fuse`, which applies
the Poisson-intensity construction *downstream*, across agents, after the features
are warped into the ego frame. The hook here applies it *upstream*, inside one
agent, before anything is transmitted.

Only the feature gateway is kept (``feature_gateway.py``, config key
``fusion_backbone.levy.gateway``, call site ``LevyPyramidFusion.forward_collab``).
The depth-pool, intra-modality and temporal hooks were removed in 371b9ec and can
be recovered from e684007.

Every hook is built by a ``build_*`` function that returns ``None`` when its config
block is absent or carries ``enable: false``, and prints one greppable line
(``[LevyHook] gateway ENABLED ...``) when it is built. In-process::

    from opencood.models import LevyHooks
    print(LevyHooks.summary())
"""

from opencood.models.LevyHooks.comm_mask import CommMask, build_comm_mask
from opencood.models.LevyHooks.common import active_hooks, reset_registry, summary
from opencood.models.LevyHooks.feature_gateway import (GatewayHook,
                                                       LevyFeatureGateway,
                                                       SigmoidFeatureGateway,
                                                       build_gateway_hook)

__all__ = [
    "CommMask",
    "build_comm_mask",
    "GatewayHook",
    "LevyFeatureGateway",
    "SigmoidFeatureGateway",
    "build_gateway_hook",
    "active_hooks",
    "reset_registry",
    "summary",
]
