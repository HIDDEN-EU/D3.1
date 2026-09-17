import numpy as np
import torch
from torch import nn
from torchvision import transforms

from opencood.models.fuse_modules.fusion_in_one import (
    AttFusion,
    DiscoFusion,
    MaxFusion,
    V2VNetFusion,
    V2XViTFusion,
)
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.lss_submodule import (
    BevEncode,
    CamEncode,
    CamEncode_Resnet101,
)
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.utils.camera_utils import (
    QuickCumsum,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class PointPillarBaseline(nn.Module):
    """
    PointPillar encoder for sensor fusion. Returns the 2D-BEV features
    """
    def __init__(self, args):
        super().__init__()
        
        grid_size = (np.array(args['lidar_range'][3:6]) - np.array(args['lidar_range'][0:3])) / \
                            np.array(args['voxel_size'])
        grid_size = np.round(grid_size).astype(np.int64)
        args['point_pillar_scatter']['grid_size'] = grid_size

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])

        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)

        # why does it call the backbone here and not in the heter_encoders.py file?
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        return spatial_features_2d


class LiftSplatShoot(nn.Module):
    """
    LiftSplatShoot for sensor fusion. Returns 2D-BEV features.
    """
    def __init__(self, args): 
        super().__init__()
        self.grid_conf = args['grid_conf']   # 网格配置参数
        self.data_aug_conf = args['data_aug_conf']   # 数据增强配置参数
        self.bevout_feature = args['bevout_feature']
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                self.grid_conf['ybound'],
                                self.grid_conf['zbound'],
                                )  # 划分网格

        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [0.4,0.4,20]
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [-49.8,-49.8,0]
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [250,250,1]
        
        self.downsample = args['img_downsample']  # 下采样倍数
        self.camC = args['img_features']  # 图像特征维度
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))  # frustum: DxfHxfWx3(41x8x16x3)

        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camera_encoder_type = args['camera_encoder']
        if self.camera_encoder_type == 'EfficientNet':
            self.camencode = CamEncode(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
        elif self.camera_encoder_type == 'Resnet101':
            self.camencode = CamEncode_Resnet101(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])

        self.bevencode = BevEncode(inC=self.camC, outC=self.bevout_feature)

        self.backbone = ResNetBEVBackbone(args['backbone_args'], 64)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        # for p in self.parameters():
        #     p.requires_grad = False
        # for p in self.camencode.depth_head.parameters():
        #     p.requires_grad = True
        #     print("freeze ",p)
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]
        
        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3) 

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        depth_items, x = self.camencode(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, depth_items

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)
        x_img, depth_items = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        x = self.voxel_pooling(geom, x_img)  # x: 4 x 64 x 240 x 240

        return x, depth_items

    def forward(self, data_dict):
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, _ = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)

        bev_features = self.bevencode(x)  # 用resnet18提取特征  x: 4 x C x 240 x 240

        return bev_features        

class CameraLidarFusion(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lift_splat_shoot_model = LiftSplatShoot(args['camera'])
        self.point_pillar_model = PointPillarBaseline(args['lidar'])

        """
        Local Multi-Modal Fusion
        """
        # lidar + cam channels
        in_channels = 512
        fused_channels = 256

        # try 1-d convolution
        self.local_fusion_block = nn.Sequential(
            nn.Conv2d(in_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        V2X Fusion
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])


        """
        Shared Heads
        """
        # self.out_channel = args['shrink_header']['dim'][-1]  \
        #                 if self.shrink_flag else fused_channels
        self.out_channel = 384

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

    # get all LiDAR and camera data for each agent
    def forward(self, data_dict):
        output_dict = {'camera_lidar_fusion':'collab'}

        # cam_bev shape: [B, 128, 256, 256]
        cam_bev = self.lift_splat_shoot_model(data_dict)
        # lidar_bev shape: [B, 384, 128, 128]
        lidar_bev = self.point_pillar_model(data_dict)

        # align camera bev features to match LiDAR size
        _, _, target_H, target_W = lidar_bev.shape
        crop_func = transforms.CenterCrop((target_H, target_W))
        cam_bev_aligned = crop_func(cam_bev)

        concat_local_features = torch.cat([lidar_bev, cam_bev_aligned], dim=1)
        enriched_local_bev = self.local_fusion_block(concat_local_features)

        if self.shrink_flag:
            transmission_bev = self.shrink_conv(enriched_local_bev)
        else:
            transmission_bev = enriched_local_bev

        # V2X alignment & multi-agent fusion
        record_len = data_dict['record_len']
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], target_H, target_W, self.point_pillar_model.voxel_size[0])

        v2x_fused_feature, _ = self.pyramid_backbone.forward_collab(
            transmission_bev,
            record_len,
            affine_matrix,
        )

        cls_preds = self.cls_head(v2x_fused_feature)
        reg_preds = self.reg_head(v2x_fused_feature)
        dir_preds = self.dir_head(v2x_fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        # output_dict.update({'occ_single_list': 
        #                     occ_outputs})


        return output_dict