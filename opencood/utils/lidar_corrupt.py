import numpy as np


def resolve_lidar_corrupt(params):
    """
    Read the ``lidar_corrupt`` block into ``(enabled, ratio)``.

    Parameters
    ----------
    params : dict
        The dataset's ``params`` (the loaded hypes).

    Returns
    -------
    (bool, float)
        ``ratio`` is meaningless when the flag is False.
    """
    cfg = params.get("lidar_corrupt") or {}
    enabled = bool(cfg.get("enabled", False))
    ratio = float(cfg.get("dropout_ratio", 0.0))
    if enabled and not (0.0 <= ratio < 1.0):
        raise ValueError(
            f"lidar_corrupt.dropout_ratio must be in [0, 1), got {ratio}. "
            "A ratio of 1.0 would drop every point."
        )
    return (enabled and ratio > 0.0), ratio


def announce_lidar_corrupt(params, where):
    """
    Print the resolved setting once, at dataset construction.

    ``where`` names the calling dataset so it is obvious that both variants now
    behave identically.
    """
    enabled, ratio = resolve_lidar_corrupt(params)
    if enabled:
        print(
            f"[lidar_corrupt] ACTIVE in {where}: dropping {ratio:.1%} of every "
            f"agent's points. Set lidar_corrupt.enabled=false in the config, or "
            f"pass --no_lidar_corrupt, for a clean run."
        )
    elif params.get("lidar_corrupt") or {}:
        print(
            f"[lidar_corrupt] present but off in {where} "
            f"(enabled={(params.get('lidar_corrupt') or {}).get('enabled')}, "
            f"dropout_ratio={(params.get('lidar_corrupt') or {}).get('dropout_ratio')})."
        )
    return enabled, ratio


def apply_lidar_corrupt(selected_cav_base, enabled, ratio, rng=None):
    """
    Drop a random ``ratio`` of one agent's points, in place.

    No-op when ``enabled`` is False or the agent carries no ``lidar_np`` (camera
    agents), so callers do not need to special-case either.

    Parameters
    ----------
    selected_cav_base : dict
        One agent's base data; mutated.
    enabled, ratio : bool, float
        As returned by :func:`resolve_lidar_corrupt`.
    rng : np.random.Generator, optional
        Supply one to make a run reproducible; defaults to the global stream,
        which is what the original inline implementation used.
    """
    if not enabled:
        return
    lidar = selected_cav_base.get("lidar_np")
    if lidar is None or len(lidar) == 0:
        return
    draw = rng.random(len(lidar)) if rng is not None else np.random.rand(len(lidar))
    selected_cav_base["lidar_np"] = lidar[draw > ratio]
