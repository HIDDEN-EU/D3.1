import torch


def bev_pool(feats, coords, B, D, H, W):
    """
    Pure PyTorch replacement for MIT BEVFusion mmdet3d.ops.bev_pool.

    Args:
        feats:
            [N, C]

        coords:
            [N, 4]
            coords[:, 0] = x index
            coords[:, 1] = y index
            coords[:, 2] = z index
            coords[:, 3] = batch index

        B:
            batch size

        D:
            number of z cells

        H:
            number of x cells

        W:
            number of y cells

    Returns:
        out:
            [B, C, D, H, W]
    """

    coords = coords.long()

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    b = coords[:, 3]

    # Flatten (b,z,x,y) into one voxel index
    linear_idx = (
        b * (D * H * W)
        + z * (H * W)
        + x * W
        + y
    )

    C = feats.shape[1]

    # [B*D*H*W, C]
    out = feats.new_zeros(
        B * D * H * W,
        C,
    )

    # Sum all features falling into the same voxel.
    out.index_add_(
        0,
        linear_idx,
        feats,
    )

    # [B,D,H,W,C]
    out = out.view(
        B,
        D,
        H,
        W,
        C,
    )

    # Match BEVFusion's output:
    # [B,C,D,H,W]
    out = out.permute(
        0, 4, 1, 2, 3
    ).contiguous()

    return out