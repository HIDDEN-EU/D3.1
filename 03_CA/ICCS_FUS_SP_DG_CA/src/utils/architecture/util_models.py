from src.utils.architecture.encoders import ConvBNReLU
import torch.nn as nn
import torch
from torch.nn import functional as F 

class BEVFusionNeck(nn.Module):
    def __init__(self, cam_channels=64, lidar_channels=64, out_channels=128):
        super().__init__()

        self.fuse = nn.Sequential(
            ConvBNReLU(cam_channels + lidar_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, cam_bev, lidar_bev):
        """
        cam_bev:   [B, C1, H, W]
        lidar_bev: [B, C2, H, W]
        """
        if cam_bev.shape[-2:] != lidar_bev.shape[-2:]:
            lidar_bev = F.interpolate(
                lidar_bev,
                size=cam_bev.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([cam_bev, lidar_bev], dim=1)
        return self.fuse(x)
    
class CenterPointHead(nn.Module):
    """
    does detection on the unified BEV features [h,w,c], c feat dimensionality. 
    Unified means c,l expressed in one coordinate system

    inputs
    in_channels: input features dimensionality 
    """
    def __init__(self, in_channels=128, num_classes=3):

        super().__init__()

        self.shared = nn.Sequential(
            ConvBNReLU(in_channels, 128),
            ConvBNReLU(128, 128),
        )

        self.heatmap_head = nn.Conv2d(128, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(128, 2, kernel_size=1)       # dx, dy
        self.height_head = nn.Conv2d(128, 1, kernel_size=1)    # z
        self.dim_head = nn.Conv2d(128, 3, kernel_size=1)       # l, w, h
        self.rot_head = nn.Conv2d(128, 2, kernel_size=1)       # sin yaw, cos yaw

    def forward(self, x):
        x = self.shared(x)

        return {
            "heatmap": torch.sigmoid(self.heatmap_head(x)),#Centre score for each class
            "reg": self.reg_head(x), #Fractional x/y offsets within the cell
            "height": self.height_head(x), #Box’s vertical centre, in metres
            "dim": self.dim_head(x), #Logarithms of length, width and height
            "rot": self.rot_head(x),#Sine and cosine of yaw
        }
    