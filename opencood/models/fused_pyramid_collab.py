"""
Homogeneous cooperative perception model where every agent uses a per-agent
fusion of PointPillar (lidar BEV) and LiftSplatShoot (camera BEV) features
before inter-agent collaborative fusion via PyramidFusion.

Based on HeterPyramidCollab by Yifan Lu <yifan_lu@sjtu.edu.cn>.
"""

import torch
import torch.nn as nn
from collections import Counter
import torchvision

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.levy_pyramid_fuse import LevyPyramidFusion
from opencood.models.heter_encoders import PointPillar, LiftSplatShoot
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn


class FusedPyramidCollab(nn.Module):
    """
    All agents share the same sensor setup: lidar + camera.

    Per-agent pipeline:
        PointPillar(lidar) → lidar_bev
        LiftSplatShoot(camera) → cam_bev  → crop
        concat([lidar_bev, cam_bev]) → 1×1 conv → fused_bev
        → backbone → aligner
    Inter-agent:
        PyramidFusion collaborative fusion → detection heads
    """

    def __init__(self, args):
        super(FusedPyramidCollab, self).__init__()
        self.args = args

        modality_name_list = [
            x for x in args.keys() if x.startswith("m") and x[1:].isdigit()
        ]
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']

        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            assert model_setting['sensor_type'] == 'lidar_camera', \
                f"FusedPyramidCollab requires sensor_type='lidar_camera', got {model_setting['sensor_type']}"

            # Lidar encoder (PointPillar)
            setattr(self, f"encoder_lidar_{modality_name}",
                    PointPillar(model_setting['encoder_lidar_args']))

            # Camera encoder (LiftSplatShoot)
            setattr(self, f"encoder_camera_{modality_name}",
                    LiftSplatShoot(model_setting['encoder_camera_args']))

            if model_setting['encoder_camera_args'].get('depth_supervision', False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # 1×1 conv to fuse concatenated lidar+camera BEV channels
            fusion_args = model_setting['fusion_args']
            setattr(self, f"fusion_conv_{modality_name}",
                    nn.Sequential(
                        nn.Conv2d(fusion_args['in_channels'], fusion_args['out_channels'],
                                  kernel_size=1, bias=False),
                        nn.BatchNorm2d(fusion_args['out_channels']),
                        nn.ReLU(inplace=True),
                    ))

            # BEV backbone
            setattr(self, f"backbone_{modality_name}",
                    ResNetBEVBackbone(model_setting['backbone_args']))

            # Feature aligner
            setattr(self, f"aligner_{modality_name}",
                    AlignNet(model_setting['aligner_args']))

            # Camera crop ratios (to match camera BEV spatial extent to lidar range)
            camera_mask_args = model_setting['camera_mask_args']
            crop_ratio_W = self.cav_range[3] / camera_mask_args['grid_conf']['xbound'][1]
            crop_ratio_H = self.cav_range[4] / camera_mask_args['grid_conf']['ybound'][1]
            setattr(self, f"crop_ratio_W_{modality_name}", crop_ratio_W)
            setattr(self, f"crop_ratio_H_{modality_name}", crop_ratio_H)

        # Spatial dimensions used for pairwise transform normalization
        self.H = self.cav_range[4] - self.cav_range[1]
        self.W = self.cav_range[3] - self.cav_range[0]
        self.fake_voxel_size = 1

        # Inter-agent multi-scale collaborative fusion
        self.pyramid_backbone = LevyPyramidFusion(args['fusion_backbone'])

        # Optional shrink header
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # Detection heads
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'], kernel_size=1)
        self.dir_head = nn.Conv2d(
            args['in_head'],
            args['dir_args']['num_bins'] * args['anchor_number'],
            kernel_size=1,
        )

        # Optional feature compressor
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(
                args['compressor']['input_dim'],
                args['compressor']['compress_ratio'],
            )

        # ------------------------------------------------------------------
        # Optional weight freezing (HEAL-style staged training).
        #
        # When this model is initialised from a pretrained HEAL stage1 lidar
        # collaboration base (heter_pyramid_collab / m1), we usually want to
        # keep that collaboration base fixed and only train the new per-agent
        # lidar+camera fused encoder so its output stays compatible with the
        # already-trained PyramidFusion + detection heads.
        #
        #   freeze_collab_base   -> pyramid_backbone, shrink_conv, {cls,reg,dir}_head
        #   freeze_lidar_encoder -> encoder_lidar_{m}, backbone_{m}
        # ------------------------------------------------------------------
        self.fix_modules = []
        if args.get('freeze_collab_base', False):
            self.fix_modules += ['pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']
            if self.shrink_flag:
                self.fix_modules.append('shrink_conv')
        if args.get('freeze_lidar_encoder', False):
            for modality_name in self.modality_name_list:
                self.fix_modules += [f'encoder_lidar_{modality_name}',
                                     f'backbone_{modality_name}']

        self.model_train_init()
        check_trainable_module(self)

    def model_train_init(self):
        """Freeze the modules listed in ``self.fix_modules`` (params + BN stats).

        ``train.py`` calls this once per epoch after ``model.train()`` so the
        frozen sub-modules never re-enter training mode.
        """
        for module_name in self.fix_modules:
            module = getattr(self, module_name)
            for p in module.parameters():
                p.requires_grad_(False)
            module.apply(fix_bn)

    # ------------------------------------------------------------------
    # Helper: forward a single modality through both encoders
    # ------------------------------------------------------------------
    def _encode_modality(self, data_dict, modality_name):
        """
        Run PointPillar + LiftSplatShoot for all agents of `modality_name`,
        crop camera BEV, concatenate, apply fusion conv.

        The data_dict contains:
            data_dict['inputs_{modality_name}']['lidar']   – lidar voxel batch
            data_dict['inputs_{modality_name}']['camera']  – camera image batch
        """
        inputs = data_dict[f'inputs_{modality_name}']

        # --- Lidar BEV ---
        encoder_lidar = getattr(self, f"encoder_lidar_{modality_name}")
        # PointPillar expects data_dict[f'inputs_{modality_name}'] = voxel dict.
        # We pass a proxy dict that maps the key it looks for.
        lidar_proxy = {f'inputs_{modality_name}': inputs['lidar']}
        lidar_bev = encoder_lidar(lidar_proxy, modality_name)  # [N, C_l, H, W]

        # --- Camera BEV ---
        encoder_camera = getattr(self, f"encoder_camera_{modality_name}")
        cam_proxy = {f'inputs_{modality_name}': inputs['camera']}
        cam_bev = encoder_camera(cam_proxy, modality_name)     # [N, C_c, H', W']

        if getattr(self, f"depth_supervision_{modality_name}"):
            # store depth items for loss computation
            pass  # encoder_camera.depth_items is set inside LiftSplatShoot.forward

        # --- Crop camera BEV to lidar spatial extent ---
        _, _, H, W = cam_bev.shape
        target_H = int(H * getattr(self, f"crop_ratio_H_{modality_name}"))
        target_W = int(W * getattr(self, f"crop_ratio_W_{modality_name}"))
        crop_fn = torchvision.transforms.CenterCrop((target_H, target_W))
        cam_bev = crop_fn(cam_bev)

        # --- Concat + 1×1 conv ---
        fused = torch.cat([lidar_bev, cam_bev], dim=1)         # [N, C_l+C_c, H, W]
        fusion_conv = getattr(self, f"fusion_conv_{modality_name}")
        fused = fusion_conv(fused)                              # [N, C_out, H, W]

        return fused

    # ------------------------------------------------------------------
    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}

        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size
        )
        record_len = data_dict['record_len']

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue

            # Per-agent dual-sensor BEV fusion
            fused_bev = self._encode_modality(data_dict, modality_name)

            # BEV backbone
            backbone = getattr(self, f"backbone_{modality_name}")
            fused_bev = backbone({'spatial_features': fused_bev})['spatial_features_2d']

            # Aligner
            aligner = getattr(self, f"aligner_{modality_name}")
            fused_bev = aligner(fused_bev)

            modality_feature_dict[modality_name] = fused_bev

            # Depth supervision items (if any)
            encoder_camera = getattr(self, f"encoder_camera_{modality_name}")
            if getattr(self, f"depth_supervision_{modality_name}"):
                output_dict[f"depth_items_{modality_name}"] = encoder_camera.depth_items

        # Assemble per-agent feature list preserving agent order
        counting_dict = {m: 0 for m in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # Inter-agent collaborative fusion.
        # cam_crop_info is left empty because all agents already have lidar-extent features
        # (the intra-agent lidar+camera fusion crops camera BEV before concatenation).
        fused_feature, occ_outputs, levy_uncertainty = self.pyramid_backbone.forward_collab(
            heter_feature_2d,
            record_len,
            affine_matrix,
            agent_modality_list,
            {},
            return_uncertainty=True,
        )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'occ_single_list': occ_outputs,
            # Levy spatial trust map: 'evidence', 'disagreement', 'sigma',
            # 'sigma_calibrated', 'coverage' at level-0 BEV resolution, plus
            # 'per_scale'. Read the first three through 'coverage': tr Sigma is
            # identically 0 wherever fewer than two agents contribute.
            # Diagnostic only -- detached by default, no effect on the loss.
            'levy_uncertainty': levy_uncertainty,
        })

        return output_dict
