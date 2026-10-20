dataset_type = 'COHFFOccDataset'
dataset_root = 'data/OPV2V/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

occ_names = [
    'building', 'fence', 'terrain', 'pole', 'road', 'sidewalk',
    'vegetation', 'vehicle', 'wall', 'guardrail', 'traffic_sign', 'bridge',
]

splits = {
    "train": [
        "2021_08_16_22_26_54",
        "2021_08_18_19_11_02",
        "2021_08_18_23_23_19",
        "2021_08_20_20_39_00",
        "2021_08_21_15_41_04",
        "2021_08_21_21_35_56",
        "2021_08_22_07_24_12",
        "2021_08_22_10_10_40",
        "2021_08_22_22_30_58",
        "2021_08_23_11_22_46",
        "2021_08_23_16_42_39",
        "2021_08_23_20_47_11",
        "2021_08_24_09_25_42",
        "2021_08_24_21_29_28",
        "2021_09_09_23_21_21",
        "2021_08_18_09_02_56",
        "2021_08_18_21_38_28",
        "2021_08_19_15_07_39",
        "2021_08_20_21_00_19",
        "2021_08_21_16_08_42",
        "2021_08_21_22_21_37",
        "2021_08_22_08_39_02",
        "2021_08_22_10_46_58",
        "2021_08_23_10_47_16",
        "2021_08_23_12_13_48",
        "2021_08_23_17_07_55",
        "2021_08_23_22_31_01",
        "2021_08_24_09_58_32",
        # "2021_09_09_13_20_58", # Excluded because there is no camera3 !!
        "2021_09_10_12_07_11",
        "2021_08_18_18_33_56",
        "2021_08_18_22_16_12",
        "2021_08_20_16_20_46",
        "2021_08_21_09_09_41",
        "2021_08_21_17_00_32",
        "2021_08_22_06_43_37",
        "2021_08_22_09_43_53",
        "2021_08_22_11_29_38",
        "2021_08_23_11_06_41",
        "2021_08_23_13_10_47",
        "2021_08_23_19_27_57",
        "2021_08_23_23_08_17",
        "2021_08_24_12_19_30",
        "2021_09_09_22_21_11",
    ],
    "validate": [
        "2021_08_20_21_48_35",
        "2021_08_21_17_30_41",
        "2021_08_22_13_37_16",
        "2021_08_22_22_01_17",
        "2021_08_23_10_51_24",
        "2021_08_23_13_17_21",
        "2021_08_23_19_42_07",
        "2021_09_09_19_27_35",
        "2021_09_11_00_33_16",
    ],
    "test": [
        "2021_08_18_19_48_05",
        "2021_08_21_09_28_12",
        "2021_08_22_09_08_29",
        "2021_08_23_12_58_19",
        "2021_08_23_16_06_26",
        "2021_08_23_21_07_10",
        "2021_08_24_07_45_41",
        "2021_08_24_20_09_18",
        "2021_08_20_21_10_24",
        "2021_08_22_07_52_02",
        "2021_08_22_21_41_24",
        "2021_08_23_15_19_19",
        "2021_08_23_17_22_47",
        "2021_08_23_21_47_19",
        "2021_08_24_11_37_54",
        "2021_08_24_20_49_54",
    ],
}

VOGS_W=[1.0117, 1.0253, 0.8993, 1.1496, 0.8413, 0.9092, 0.9600, 0.9374, 1.0025, 1.0226, 1.4045, 1.3744]
NUSC_W=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1]

VOGS_W_MEAN = sum(VOGS_W) / len(VOGS_W)
NUSC_W_MEAN = sum(NUSC_W) / len(NUSC_W)
SCALE = NUSC_W_MEAN / VOGS_W_MEAN

COHFF_W = [w * SCALE for w in VOGS_W]

assert len(COHFF_W) == len(occ_names), "Length of COHFF_W must match length of occ_names"

point_cloud_range = [-20.0, -20.0, -2.3, 20.0, 20.0, 0.9]
occ_size = [100, 100, 8]
voxel_size = [
    (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0],
    (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1],
    (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2],
]

embed_dims = 256
num_layers = 5
num_query = 600
num_frames = 1
num_views = 4
num_levels = 4
num_points = 4
num_refines = [1, 4, 8, 16, 32]
# num_refines = [1, 2, 4, 8, 16]
num_pt_channels = 32

img_backbone = dict(
    type='ResNet',
    depth=101,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(
        type='BN2d',
        requires_grad=False,
        ),
    norm_eval=True,
    style='caffe',
    with_cp=True,
    dcn=dict(
        type='DCNv2',
        deform_groups=1,
        fallback_on_stride=False),
    stage_with_dcn=(False, False, True, True),
    )

img_neck = dict(
    type='FPN',
    start_level=1,
    out_channels=embed_dims,
    in_channels=[256, 512, 1024, 2048],
    num_outs=num_levels,
    add_extra_convs='on_output',
    relu_before_extra_convs=True,
    )

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

model = dict(
    type='OPUSV2',
    use_grid_mask=True,
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='OPUSV2Head',
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        pc_range=point_cloud_range,
        empty_label=len(occ_names),
        voxel_size=voxel_size,
        mask_filter=True,  # exclude out-of-taxonomy voxels from the loss (not from gt geometry/matching)
        transformer=dict(
            type='OPUSV2Transformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_views=num_views,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_refines=num_refines,
            num_pt_channels=num_pt_channels,
            scales=[0.5],
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5)),
    train_cfg=dict(
        # adjust class weights TODO
        pts=dict(
            # cls_weights=[
            #     1.5, 2, 4, 4, 1, 1, 1, 1, 2, 1, 4, 4
            # ],
            #cls_weights=[1.0] * len(occ_names),
            cls_weights=COHFF_W,
            rare_weights_scale=2,  # CoHFF doubles the default rare-class weight (10 -> 20)
            rare_classes=[
                occ_names.index('pole'),
                occ_names.index('guardrail'),
                occ_names.index('traffic_sign'),
                occ_names.index('bridge'),
            ])),
    test_cfg=dict(pts=dict(score_thr=0.35)),
)

ida_aug_conf = dict(
    resize_lim=(0.8, 1.2),
    final_dim=(576, 800),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0, 0),
    H=600,
    W=800,
    rand_flip=True,
)

bda_aug_conf = dict(
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

camera_names = ['camera0', 'camera1', 'camera2', 'camera3']
train_pipeline = [
    dict(
        type='LoadCOHFFMVImageWithSweeps',
        camera_names=camera_names,
        prev_sweeps_num=num_frames - 1),
    dict(
        type='LoadCOHFFOccupancyAsOPUS',
        grid_size=occ_size,
        pc_range=point_cloud_range,
        empty_label=len(occ_names),
        with_mask_camera=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='RandomTransformOcc', bda_aug_conf=bda_aug_conf),
    dict(
        type='FinalFormatting',
        keys=['img', 'voxel_semantics', 'mask_camera'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2occ',
            'ego2img', 'ego2lidar', 'img_timestamp', 'scene_name',
            'sample_token', 'frame_id', 'agent', 'num_frames', 'num_views')),
]

test_pipeline = [
    dict(
        type='LoadCOHFFMVImageWithSweeps',
        camera_names=camera_names,
        prev_sweeps_num=num_frames - 1,
        test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='FinalFormatting',
        test_mode=True,
        keys=['img'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2occ',
            'ego2img', 'ego2lidar', 'img_timestamp', 'scene_name',
            'sample_token', 'frame_id', 'agent', 'num_frames', 'num_views')),
]

# Evaluate only on frames in which the ego has a neighbor, i.e. the same samples
# as the collaborative protocol (distance / overlap as in the collab config).
neighbor_filter = dict(
    max_agent_distance=70.0,
    min_bev_overlap=0.05,
    overlap_grid_resolution=40)

data = dict(
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        split='train',
        splits=splits,
        pipeline=train_pipeline,
        classes=occ_names,
        camera_names=camera_names,
        test_mode=False,
        info_cache_file='data/cohff_opusv2_train_infos.pkl',
        rebuild_cache=False,
        use_merged_gt=False,  # True: train/eval against merged multi-agent GT
        ego_perspectives='first', # 'first' or 'all'
        occ_size=occ_size,
        pc_range=point_cloud_range,
        empty_label=len(occ_names)),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        split='validate',
        splits=splits,
        pipeline=test_pipeline,
        classes=occ_names,
        camera_names=camera_names,
        test_mode=True,
        info_cache_file='data/cohff_opusv2_val_infos.pkl',
        rebuild_cache=False,
        neighbor_filter=neighbor_filter,
        use_merged_gt=False,  # True: train/eval against merged multi-agent GT
        ego_perspectives='first', # 'first' or 'all'
        occ_size=occ_size,
        pc_range=point_cloud_range,
        empty_label=len(occ_names)),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        split='test',
        splits=splits,
        pipeline=test_pipeline,
        classes=occ_names,
        camera_names=camera_names,
        test_mode=True,
        info_cache_file='data/cohff_opusv2_test_infos.pkl',
        rebuild_cache=False,
        neighbor_filter=neighbor_filter,
        use_merged_gt=False,
        ego_perspectives='first',  # 'first' or 'all'
        occ_size=occ_size,
        pc_range=point_cloud_range,
        empty_label=len(occ_names)),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        # 'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 30
batch_size = 8

# R101+DCN pretrain, matching the backbone (nuScenes-Occ3D's V2 template
# points at an R50 checkpoint, which wouldn't load into this backbone).
load_from = "pretrain/r101_dcn_fcos3d_pretrain.pth"
revise_keys = None
resume_from = None
checkpoint_config = dict(interval=5, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval=50, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ])
eval_config = dict(interval=5)
debug = False
