import os
import re

import mmcv
import numpy as np
import yaml
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

from models.utils import sparse2dense
from .old_metrics import Metric_mIoU_Occ3D
from .pipelines.cohff_loading import LoadCOHFFOccupancyAsOPUS
from .pipelines.cohff_loading import build_opus_label_map
from .custom import CustomDataset


def _load_and_voxelize_gt(args):
    """Module-level worker so mmcv.track_parallel_progress can pickle it."""
    pcd_path, grid_size, pc_range, empty_label, label_map = args
    loader = LoadCOHFFOccupancyAsOPUS(
        grid_size=grid_size,
        pc_range=pc_range,
        empty_label=empty_label,
        label_map=label_map,
        with_mask_camera=True,
    )
    results = loader({'gt_sem_pc_path': pcd_path})
    return results['voxel_semantics'], results['mask_camera']


def pose_to_world(pose):
    """Convert OPV2V [x, y, z, roll, yaw, pitch] to a world transform."""
    x, y, z, roll, yaw, pitch = map(float, pose[:6])
    roll, pitch, yaw = np.radians([roll, pitch, yaw])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [x, y, z]
    transform[:3, :3] = [
        [cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return transform


@DATASETS.register_module()
class COHFFOccDataset(CustomDataset):
    """Single-agent CoHFF (OPV2V) occupancy dataset.

    With use_merged_gt=True, the GT of each ego is read from the precomputed
    merged semantic occupancy (<surround>/<split>/<scene>/<merged_gt_dirname>/
    <frame>_ego<id>_merged_semantic.npz), for training and evaluation alike.
    With neighbor_filter=dict(max_agent_distance, min_bev_overlap,
    overlap_grid_resolution), only samples that have at least one neighbor
    agent in the same frame are kept (the evaluation set of the collaborative
    protocol). Every agent of a scene is treated as an independent ego vehicle
    (ego_perspectives='all'), or only the first agent of each scene
    (ego_perspectives='first').
    """
    CLASSES = ()

    def __init__(self,
                 split,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 camera_names=None,
                 splits=None,
                 load_interval=1,
                 test_mode=False,
                 max_samples=None,
                 info_cache_file=None,
                 rebuild_cache=False,
                 use_merged_gt=False,
                 merged_gt_dirname='merged_gt',
                 neighbor_filter=None,
                 ego_perspectives='first',
                 occ_size=(100, 100, 8),
                 pc_range=(-20, -20, -2.3, 20, 20, 0.9),
                 empty_label=0,
                 label_map=None,
                 **kwargs):
        super().__init__()
        if ego_perspectives not in ('all', 'first'):
            raise ValueError(
                f"Invalid ego_perspectives: {ego_perspectives}. "
                "Must be 'all' or 'first'.")
        self.split = split
        self.data_root = data_root
        self.camera_names = camera_names or [
            'camera0', 'camera1', 'camera2', 'camera3']
        self.splits = splits
        self.load_interval = load_interval
        self.test_mode = test_mode
        self.filter_empty_gt = False  # read by CustomDataset
        self.max_samples = max_samples
        self.CLASSES = self.get_classes(classes)
        self.rebuild_cache = rebuild_cache
        self.info_cache_file = info_cache_file
        self.empty_label = empty_label
        self.pc_range = np.asarray(pc_range, dtype=np.float32)
        self.ego_perspectives = ego_perspectives
        self.use_merged_gt = use_merged_gt
        self.neighbor_filter = neighbor_filter
        self.merged_gt_dirname = merged_gt_dirname

        all_infos = self.load_annotations(None)

        # Per-agent timelines, used for temporal sweeps
        self.timeline_positions = {}
        timelines = {}
        for info in all_infos:
            timelines.setdefault(
                (info['scene_name'], info['agent']), []).append(info)
        for timeline in timelines.values():
            timeline.sort(key=lambda x: x['timestamp'])
            for position, info in enumerate(timeline):
                self.timeline_positions[info['token']] = (timeline, position)

        # Sample selection: (1) merged GT available, (2) a neighbor exists,
        # (3) ego agent(s) per scene chosen among the remaining candidates.
        candidates = [info for info in all_infos if int(info['agent']) >= 0]
        if self.use_merged_gt:
            candidates = [
                info for info in candidates
                if self._attach_merged_gt_path(info)]
        if self.neighbor_filter is not None:
            candidates = self._filter_by_neighbors(candidates, all_infos)

        agents_by_scene = {}
        for info in candidates:
            agents_by_scene.setdefault(
                info['scene_name'], set()).add(info['agent'])
        if ego_perspectives == 'first':
            selected = {
                scene: {min(agents, key=lambda a: (int(a), str(a)))}
                for scene, agents in agents_by_scene.items()}
        else:
            selected = agents_by_scene
        self.data_infos = [
            info for info in candidates
            if info['agent'] in selected[info['scene_name']]]

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        if not self.test_mode:
            self._set_group_flag()

        self.label_mapping = (
            label_map if label_map is not None
            else build_opus_label_map(self.empty_label))
        self.occ_loader = LoadCOHFFOccupancyAsOPUS(
            grid_size=occ_size,
            pc_range=pc_range,
            empty_label=empty_label,
            label_map=self.label_mapping,
            with_mask_camera=True,
        )
        # Caching instead of costly revoxelization
        self._gt_occ_cache = {}

    def _bev_overlap(self, ego_info, other_info, resolution):
        """Fraction of the ego's BEV grid centers that lie inside the other
        agent's BEV range."""
        x_min, y_min = self.pc_range[:2]
        x_max, y_max = self.pc_range[3:5]
        xs = x_min + (np.arange(resolution) + 0.5) * (x_max - x_min) / resolution
        ys = y_min + (np.arange(resolution) + 0.5) * (y_max - y_min) / resolution
        xv, yv = np.meshgrid(xs, ys, indexing='ij')
        ego_points = np.stack([
            xv.reshape(-1), yv.reshape(-1),
            np.zeros(resolution ** 2, dtype=np.float32),
            np.ones(resolution ** 2, dtype=np.float32)], axis=-1)

        other_from_ego = (
            np.linalg.inv(other_info['world_from_lidar'])
            @ ego_info['world_from_lidar']).astype(np.float32)
        pts = ego_points @ other_from_ego.T
        inside = ((pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
                  (pts[:, 1] >= y_min) & (pts[:, 1] < y_max))
        return float(inside.mean())

    def _filter_by_neighbors(self, candidates, all_infos):
        """Keep samples for which at least one other agent of the same frame
        is within `max_agent_distance` and overlaps by `min_bev_overlap`.

        This is the sample set of the collaborative CoHFF protocol, so single-
        and multi-agent models can be compared on identical samples."""
        cfg = self.neighbor_filter
        max_dist = cfg.get('max_agent_distance', 28.0)
        min_overlap = cfg.get('min_bev_overlap', 0.2)
        resolution = cfg.get('overlap_grid_resolution', 40)

        frames = {}
        for info in all_infos:
            if int(info['agent']) >= 0:
                frames.setdefault(
                    (info['scene_name'], info['frame_id']), []).append(info)

        def has_neighbor(ego):
            ego_pos = ego['world_from_lidar'][:2, 3]
            for other in frames[(ego['scene_name'], ego['frame_id'])]:
                if other['agent'] == ego['agent']:
                    continue
                dist = np.linalg.norm(
                    ego_pos - other['world_from_lidar'][:2, 3])
                if (dist <= max_dist and
                        self._bev_overlap(ego, other, resolution) >= min_overlap):
                    return True
            return False

        return [info for info in candidates if has_neighbor(info)]

    def _attach_merged_gt_path(self, info):
        merged_gt_path = os.path.join(
            self.data_root, 'surround', self.split, info['scene_name'],
            self.merged_gt_dirname,
            f"{info['frame_id']}_ego{int(info['agent'])}_merged_semantic.npz")
        if not os.path.isfile(merged_gt_path):
            return False
        info['raw_gt_sem_pc_path'] = info['gt_sem_pc_path']
        info['gt_sem_pc_path'] = merged_gt_path
        return True

    def load_annotations(self, ann_file=None):
        if (self.info_cache_file and os.path.exists(self.info_cache_file)
                and not self.rebuild_cache):
            return mmcv.load(self.info_cache_file)[::self.load_interval]

        data_infos = []
        for scene in self.splits[self.split]:
            scene_dir = os.path.join(self.data_root, self.split, scene)
            surround_dir = os.path.join(
                self.data_root, 'surround', self.split, scene)
            if not os.path.isdir(scene_dir):
                continue

            agents = [a for a in sorted(os.listdir(scene_dir))
                      if not a.endswith('.yaml')]
            for agent in agents:
                agent_dir = os.path.join(scene_dir, agent)
                if not os.path.isdir(agent_dir):
                    continue

                frame_ids = self.get_frame_ids(agent_dir)[::self.load_interval]
                for frame_id in frame_ids:
                    info = self.build_frame_info(
                        scene_dir, surround_dir, scene, agent, frame_id)
                    if info is not None:
                        data_infos.append(info)
                        if self.max_samples and len(data_infos) >= self.max_samples:
                            return data_infos

        if self.info_cache_file:
            mmcv.mkdir_or_exist(os.path.dirname(self.info_cache_file))
            mmcv.dump(data_infos, self.info_cache_file)

        return data_infos[::self.load_interval]

    def build_frame_info(self, scene_dir, surround_dir, scene, agent, frame_id):
        meta_path = os.path.join(scene_dir, agent, f'{frame_id}.yaml')
        gt_sem_pc_path = os.path.join(
            surround_dir, agent, f'{frame_id}_semantic.pcd')
        if not (os.path.exists(meta_path) and os.path.exists(gt_sem_pc_path)):
            return None
        meta = self.load_meta(meta_path)

        pose = meta.get('lidar_pose', meta.get('true_ego_pos'))
        if pose is None:
            raise KeyError(
                f"Could not find 'lidar_pose' or 'true_ego_pos' in {meta_path}")
        world_from_lidar = pose_to_world(pose)

        cams = {}
        for cam_name in self.camera_names:
            img_path = os.path.join(
                scene_dir, agent, f'{frame_id}_{cam_name}.png')
            if not os.path.exists(img_path):
                return None
            intrinsic, cam2lidar, lidar2img = self.load_camera_mats(
                meta, cam_name)
            cams[cam_name] = dict(
                data_path=img_path,
                timestamp=self.frame_id_to_timestamp(frame_id),
                cam_intrinsic=intrinsic,
                cam2lidar=cam2lidar,
                ego2img=lidar2img,
                world_from_lidar=world_from_lidar,
                is_key_frame=True,
            )

        return dict(
            token=f'{scene}_{agent}_{frame_id}',
            scene_name=scene,
            agent=agent,
            frame_id=frame_id,
            timestamp=self.frame_id_to_timestamp(frame_id),
            cams=cams,
            world_from_lidar=world_from_lidar,
            gt_sem_pc_path=gt_sem_pc_path,
        )

    def get_frame_ids(self, agent_dir):
        return [f.split('_camera0')[0] for f in sorted(os.listdir(agent_dir))
                if f.endswith('camera0.png')]

    def load_meta(self, meta_path):
        with open(meta_path, 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)

    def load_camera_mats(self, meta, cam_name):
        intrinsic = np.asarray(meta[cam_name]['intrinsic'], dtype=np.float32)
        cam2lidar = np.asarray(meta[cam_name]['extrinsic'], dtype=np.float32)

        if cam2lidar.shape != (4, 4):
            cam2lidar_4x4 = np.eye(4, dtype=np.float32)
            cam2lidar_4x4[:3, :3] = cam2lidar[:3, :3]
            cam2lidar_4x4[:3, 3] = cam2lidar[:3, 3]
            cam2lidar = cam2lidar_4x4

        ue4_to_opencv = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        cam2lidar = cam2lidar @ ue4_to_opencv

        cam2img = np.eye(4, dtype=np.float32)
        cam2img[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img = cam2img @ np.linalg.inv(cam2lidar)
        return intrinsic, cam2lidar.astype(np.float32), lidar2img.astype(np.float32)

    def frame_id_to_timestamp(self, frame_id):
        match = re.search(r'\d+', str(frame_id))
        return float(match.group(0)) if match else 0.0

    def collect_cam_sweeps(self, info, into_past=75, into_future=0):
        timeline, position = self.timeline_positions[info['token']]
        prev_sweeps = [
            timeline[i]['cams']
            for i in range(position - 1, max(position - 1 - into_past, -1), -1)]
        next_sweeps = [
            timeline[i]['cams']
            for i in range(position + 1, min(position + 1 + into_future, len(timeline)))]
        return prev_sweeps, next_sweeps

    def get_data_info(self, index):
        info = self.data_infos[index]
        identity = np.eye(4, dtype=np.float32)
        cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(info)
        return dict(
            sample_token=info['token'],
            scene_name=info['scene_name'],
            agent=info['agent'],
            frame_id=info['frame_id'],
            timestamp=info['timestamp'],
            ego2lidar=identity.copy(),
            ego2occ=identity.copy(),
            world_from_lidar=info['world_from_lidar'],
            cam_info=info['cams'],
            cam_sweeps=dict(prev=cam_sweeps_prev, next=cam_sweeps_next),
            gt_sem_pc_path=info['gt_sem_pc_path'],
        )

    def load_gt_occ_from_pcd(self, pcd_path):
        if pcd_path not in self._gt_occ_cache:
            results = self.occ_loader({'gt_sem_pc_path': pcd_path})
            self._gt_occ_cache[pcd_path] = (
                results['voxel_semantics'], results['mask_camera'])
        return self._gt_occ_cache[pcd_path]

    def evaluate(self, results, logger=None, **eval_kwargs):
        return self.eval_miou(occ_results=results, **eval_kwargs)

    def eval_miou(self, occ_results, num_workers=None, **eval_kwargs):
        print('Evaluating mIoU...')

        num_semantic_classes = len(self.CLASSES)
        num_classes = num_semantic_classes + 1
        assert self.empty_label == num_semantic_classes

        needed_paths = [
            self.data_infos[i]['gt_sem_pc_path'] for i in range(len(occ_results))
            if self.data_infos[i]['gt_sem_pc_path'] not in self._gt_occ_cache
        ]
        if needed_paths:
            nproc = num_workers or min(os.cpu_count() or 1, 4)
            print(f'Loading {len(needed_paths)} GT occupancy grids using {nproc} workers...')
            worker_args = [
                (path, self.occ_loader.grid_size, self.occ_loader.pc_range,
                 self.empty_label, self.label_mapping)
                for path in needed_paths
            ]
            gt_results = mmcv.track_parallel_progress(
                _load_and_voxelize_gt, worker_args, nproc)
            for path, gt in zip(needed_paths, gt_results):
                self._gt_occ_cache[path] = gt

        metric = Metric_mIoU_Occ3D(
            num_classes=num_classes, use_image_mask=True)

        from tqdm import tqdm
        for i, result_dict in tqdm(enumerate(occ_results), total=len(occ_results)):
            gt, mask = self.load_gt_occ_from_pcd(
                self.data_infos[i]['gt_sem_pc_path'])
            pred, _ = sparse2dense(
                result_dict['occ_loc'], result_dict['sem_pred'],
                dense_shape=gt.shape, empty_value=self.empty_label)
            metric.add_batch(pred, gt, mask, mask)

        per_class_iou = metric.per_class_iu(metric.hist)
        for class_id, class_name in enumerate(self.CLASSES):
            print(f"{class_name}: {per_class_iou[class_id] * 100:.2f}%")

        miou = round(float(np.nanmean(per_class_iou[:num_semantic_classes]) * 100), 2)
        print(f"mIoU: {miou:.2f}%")
        return {"mIoU": miou}
