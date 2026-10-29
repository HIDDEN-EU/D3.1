import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

class PillarsToBEV(nn.Module):
    def __init__(self):
        super().__init__()

        voxel_size = [0.5, 0.5, 20.0]
        point_cloud_range = [-50, -50, -10, 50, 50, 10]

        self.preprocessor = Det3DDataPreprocessor(
            voxel=True,
            voxel_layer=dict(
                max_num_points=32,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=(16000, 16000),
            ),
        )
        self.voxel_encoder = MODELS.build(dict(
            type="PillarFeatureNet",
            in_channels=4,
            feat_channels=[64],
            with_distance=False,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        ))
        self.scatter = MODELS.build(dict(
            type="PointPillarsScatter",
            in_channels=64,
            output_shape=[200, 200],  # [y cells, x cells]
        ))

    def forward(self, points):
        # Your current batch is [B, N, 4]. MMDetection3D expects
        # one [Ni, 4] tensor per sample.
        if isinstance(points, torch.Tensor):
            points = list(points.unbind(0))

        vox = self.preprocessor.voxelize(points, data_samples=None)
        features = self.voxel_encoder(
            vox["voxels"], vox["num_points"], vox["coors"]
        )
        return self.scatter(features, vox["coors"], len(points))
        # [B, 64, 200, 200]