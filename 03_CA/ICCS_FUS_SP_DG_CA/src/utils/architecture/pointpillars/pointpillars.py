import torch.nn as nn

from pointpillars.model.pointpillars import (
    PillarLayer,
    PillarEncoder,
    Backbone,
    Neck,
)


class PointPillarsEncoder(nn.Module):
    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points=32,
        max_voxels=(16000, 40000),
    ):
        super().__init__()

        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channel=9,
            out_channel=64,
        )

        self.backbone = Backbone(
            in_channel=64,
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
        )

        self.neck = Neck(
            in_channels=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            out_channels=[128, 128, 128],
        )

    def forward(self, points):
        """
        points:
            list of B tensors

            each:
            [N_i, 4] = x, y, z, intensity
        """

        pillars, coors_batch, npoints_per_pillar = (
            self.pillar_layer(points)
        )

        pseudo_img = self.pillar_encoder(
            pillars,
            coors_batch,
            npoints_per_pillar,
        )

        # [B, 64, H, W]

        backbone_feats = self.backbone(
            pseudo_img
        )

        # list:
        # [B, 64,   H/2, W/2]
        # [B, 128,  H/4, W/4]
        # [B, 256,  H/8, W/8]

        bev = self.neck(backbone_feats)

        # [B, 384, H/2, W/2]

        return bev