import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LSSCameraEncoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 bev_channels=64, 
                 bev_h=200, 
                 bev_w=200):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.backbone = nn.Sequential(
            ConvBNReLU(in_channels, 32, 7, 2, 3),
            ConvBNReLU(32, 64, 3, 2, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, bev_channels, 3, 1, 1),
        )

        self.bev_proj = nn.Sequential(
            ConvBNReLU(bev_channels, bev_channels),
            ConvBNReLU(bev_channels, bev_channels),
        )

    def forward(self, img):
        """
        img: [B, 3, H, W]
        returns camera_bev: [B, C, bev_h, bev_w]
        """
        x = self.backbone(img)
        x = F.interpolate(x, size=(self.bev_h, self.bev_w), 
                          mode="bilinear", align_corners=False)
        x = self.bev_proj(x)
        return x
    


class PointPillarsEncoder(nn.Module):
    def __init__(self, in_channels=9, bev_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            ConvBNReLU(in_channels, 64),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128),
            ConvBNReLU(128, bev_channels),
        )

    def forward(self, lidar_pseudo_img):
        """
        lidar_pseudo_img: [B, C_in, H, W]
        returns lidar_bev: [B, C, H/2, W/2]
        """
        # lidar_pseudo_img = self.pillars_to_bev(lidar_pseudo_img)
        infra_features_bev = self.net(lidar_pseudo_img)
        return infra_features_bev