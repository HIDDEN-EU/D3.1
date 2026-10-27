import torch


def transform_vehicle_grid_to_infra(
    grid_vehicle,
    R_inf_to_veh,
    t_inf_to_veh,
):
    """
    Convert vehicle-frame BEV coordinates into infrastructure-frame
    coordinates.

    Given:

        p_vehicle = R_inf_to_veh @ p_infra + t_inf_to_veh

    grid_sample needs the inverse:

        p_infra = R^T @ (p_vehicle - t)

    Args:
        grid_vehicle: [H, W, 2]
        R_inf_to_veh: [3, 3]
        t_inf_to_veh: [3] or [3, 1]

    Returns:
        grid_infra: [H, W, 2]
    """

    device = grid_vehicle.device
    dtype = grid_vehicle.dtype

    R = torch.as_tensor(
        R_inf_to_veh,
        device=device,
        dtype=dtype,
    )

    t = torch.as_tensor(
        t_inf_to_veh,
        device=device,
        dtype=dtype,
    ).reshape(3)

    H, W, _ = grid_vehicle.shape

    # Vehicle BEV points -> 3D points with z = 0
    zeros = torch.zeros(
        H,
        W,
        1,
        device=device,
        dtype=dtype,
    )

    p_vehicle = torch.cat(
        [grid_vehicle, zeros],
        dim=-1,
    )  # [H, W, 3]

    p_vehicle = p_vehicle.reshape(-1, 3)

    # Inverse transformation:
    #
    # p_vehicle = R p_infra + t
    #
    # => p_infra = R^T (p_vehicle - t)

    p_infra = (p_vehicle - t) @ R

    p_infra = p_infra.reshape(H, W, 3)

    return p_infra[..., :2]