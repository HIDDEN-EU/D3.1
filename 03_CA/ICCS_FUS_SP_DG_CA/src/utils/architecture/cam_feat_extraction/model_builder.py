# from mmdet3d.models.builder import build_backbone, build_neck
from mmdet3d.registry import MODELS
from src.utils.architecture.cam_feat_extraction.gen_lss_fpn import GeneralizedLSSFPN
# import projects.BEVFusion.bevfusion

class mmdet3d_model_builder():
    def __init__(self) -> None:
        self.camera_backbone = MODELS.build(
                            dict(
                                type="mmdet.SwinTransformer",
                                embed_dims=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                attn_drop_rate=0.0,
                                drop_path_rate=0.2,
                                patch_norm=True,
                                out_indices=[1, 2, 3],
                                with_cp=False,
                                convert_weights=True,
                                init_cfg=dict(
                                    type="Pretrained",
                                    checkpoint=(
                                        "https://github.com/SwinTransformer/storage/releases/"
                                        "download/v1.0.0/swin_tiny_patch4_window7_224.pth"
                                    ),
                                ),
                            )
                            
                        )

        # self.camera_neck = MODELS.build(
        #                     dict(
        #                         type="GeneralizedLSSFPN",
        #                         in_channels=[192, 384, 768],
        #                         out_channels=256,
        #                         start_level=0,
        #                         num_outs=3,
        #                         norm_cfg=dict(
        #                             type="BN2d",
        #                             requires_grad=True,
        #                         ),
        #                         act_cfg=dict(
        #                             type="ReLU",
        #                             inplace=True,
        #                         ),
        #                         upsample_cfg=dict(
        #                             mode="bilinear",
        #                             align_corners=False,
        #                         ),
        #                     )
        #                 )
        self.camera_neck = GeneralizedLSSFPN(in_channels=[192, 384, 768],
                                            out_channels=256,
                                            num_outs=3
                                            )

        pass
    
    def get_camera_backbone(self):
        return self.camera_backbone
    
    def get_camera_neck(self):
        return self.camera_neck