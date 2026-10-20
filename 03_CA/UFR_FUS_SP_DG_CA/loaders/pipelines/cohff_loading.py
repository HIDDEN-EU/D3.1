import os.path

import mmcv
import numba as nb
import numpy as np
from mmdet.datasets.builder import PIPELINES

VQSOP_LABEL_MAP = {
    0: 0, # unlabeled
    1: 1, # Building
    2: 2, # Fence
    3: 0, # unlabeled
    4: 0, # unlabeled
    5: 4, # Pole
    6: 5, # Road
    7: 5, # Road
    8: 6, # SideWalk
    9: 7, # Vegetation
    10: 8,  # Vehicles
    11: 9, # Wall
    12: 11, # TrafficSign
    13: 0, # unlabeled
    14: 3, # Ground
    15: 12, # Bridge
    16: 0, # unlabeled
    17: 10, # GuardRail
    18: 4, # Pole
    19: 0, # unlabeled
    20: 8, # Vehicles
    21: 0, # unlabeled
    22: 3 # Terrain
    }

def build_opus_label_map(empty_label=0):
    return {
        raw_label: (
            empty_label if vqsop_label == 0 else vqsop_label - 1)
        for raw_label, vqsop_label in VQSOP_LABEL_MAP.items()}

@PIPELINES.register_module()
class LoadCOHFFOccupancyAsOPUS:

    def __init__(self,
                 grid_size,
                 pc_range,
                 empty_label=0,
                 label_map=None,
                 with_mask_camera=False):
        self.grid_size = np.array(grid_size, dtype=np.int64)
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.empty_label = empty_label
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.with_mask_camera = with_mask_camera
        # Sentinel for out-of-taxonomy objects, so that their majority
        # vote doesn't get silently counted as free space.
        self.unknown_label = 255
        self.label_mapping = (
            label_map
            if label_map is not None
            else build_opus_label_map(empty_label=self.empty_label)
            )

    def __call__(self, results):
        gt_path = results['gt_sem_pc_path']

        if str(gt_path).endswith('.npz'):
            semantics, mask_camera = self.load_merged_occupancy(gt_path)
            results['voxel_semantics'] = semantics
            if self.with_mask_camera:
                results['mask_camera'] = mask_camera
                results['mask_lidar'] = np.ones_like(
                    mask_camera, dtype=np.bool_)
            return results

        sem_pc = self.load_semantic_pointcloud(gt_path)

        raw_labels = sem_pc[:, 3].astype(np.int64)
        points = sem_pc[:, :3].astype(np.float32)
        spatial_mask = ((points >= self.pc_range[:3]) &
                        (points < self.pc_range[3:])).all(axis=1)
        points = points[spatial_mask]
        labels = self.remap_labels(raw_labels[spatial_mask]).reshape(-1, 1)

        semantics = np.ones(self.grid_size, dtype=np.uint8) * self.empty_label
        if points.shape[0] > 0:
            voxel_coords = np.floor(
                (points - self.pc_range[:3][None, :]) /
                self.voxel_size[None, :]).astype(np.int64)
            voxel_label_pairs = np.concatenate([voxel_coords, labels], axis=1)
            order = np.lexsort((
                voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]))
            semantics = nb_process_label(semantics, voxel_label_pairs[order])

        # Voxel with majority-vote unknown: flagged as invalid
        # --> mapped to empty but ignored in loss
        mask_camera = semantics != self.unknown_label
        semantics[~mask_camera] = self.empty_label

        results['voxel_semantics'] = semantics
        if self.with_mask_camera:
            results['mask_camera'] = mask_camera
            # Dummy placeholder for lidar
            results['mask_lidar'] = np.ones_like(mask_camera)

        return results

    def load_merged_occupancy(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)

        with np.load(npz_path, allow_pickle=False) as data:
            if "semantics" not in data.files:
                raise ValueError(f"Missing 'semantics' in {npz_path}")
            semantics = data["semantics"].astype(np.uint8, copy=True)
            stored_empty_label = (
                    int(data["empty_label"])
                    if "empty_label" in data.files
                    else self.empty_label)
            unknown_label = (
                    int(data["unknown_label"])
                    if "unknown_label" in data.files
                    else 13)

        expected_shape = tuple(self.grid_size.tolist())
        if semantics.shape != expected_shape:
            raise ValueError(
                f"Expected semantics shape {expected_shape}, got {semantics.shape} in {npz_path}")
        if stored_empty_label != self.empty_label:
            raise ValueError(
                f"Expected empty_label {self.empty_label}, got {stored_empty_label} in {npz_path}")

        mask_camera = semantics != unknown_label
        semantics[~mask_camera] = self.empty_label

        return semantics, mask_camera

    def load_semantic_pointcloud(self, pcd_path):
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(pcd_path)

        header_lines = 0
        data_line = None
        with open(pcd_path, 'r') as f:
            for line in f:
                header_lines += 1
                stripped = line.strip()
                if stripped.startswith('DATA'):
                    data_line = stripped
                    break

        if data_line != 'DATA ascii':
            raise ValueError(
                f"Unsupported PCD DATA format in {pcd_path}: {data_line}")

        pcd = np.loadtxt(pcd_path, skiprows=header_lines, dtype=np.float32)
        if pcd.ndim == 1:
            pcd = pcd[None, :]
        if pcd.shape[1] != 4:
            raise ValueError(
                f'Expected 4 PCD columns (x, y, z, label), got {pcd.shape[1]}')
        return pcd

    def remap_labels(self, labels):
        remapped = np.full_like(labels, self.unknown_label, dtype=np.uint8)
        for raw_id, mapped_id in self.label_mapping.items():
            remapped[labels == raw_id] = mapped_id
        return remapped

@PIPELINES.register_module()
class LoadCOHFFMVImageWithSweeps:

    def __init__(self,
                 camera_names=None,
                 prev_sweeps_num=0,
                 next_sweeps_num=0,
                 color_type='color',
                 test_mode=False):
        self.camera_names = camera_names or [
            'camera0', 'camera1', 'camera2', 'camera3'
        ]

        for name, value in (
            ('prev_sweeps_num', prev_sweeps_num),
            ('next_sweeps_num', next_sweeps_num)):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value}")

        if not self.camera_names:
            raise ValueError("camera_names must be a non-empty list of strings")

        self.prev_sweeps_num = prev_sweeps_num
        self.next_sweeps_num = next_sweeps_num
        self.num_frames = 1 + self.prev_sweeps_num + self.next_sweeps_num
        self.num_views = len(self.camera_names)
        self.color_type = color_type
        self.test_mode = test_mode

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def padding(self, values, tgt_len):
        if len(values) == 0:
            return [None for _ in range(tgt_len)]
        return values + [values[-1]] * (tgt_len - len(values))

    def extract_info(self, data, current_world_from_lidar):
        filenames, timestamps, ego2img = [], [], []

        current_world_from_lidar = np.array(current_world_from_lidar, dtype=np.float32)
        if current_world_from_lidar.shape != (4, 4):
            raise ValueError(
                f"current_world_from_lidar must be a 4x4 matrix, got shape {current_world_from_lidar.shape}")

        for cam_name in self.camera_names:
            if cam_name not in data:
                raise KeyError(f"Camera name '{cam_name}' not found in data")
            cam = data[cam_name]
            filenames.append(cam['data_path'])
            timestamps.append(cam['timestamp'])
            historical_world_from_lidar = np.asarray(
                cam['world_from_lidar'], dtype=np.float32)
            historical_ego2img = np.asarray(cam['ego2img'], dtype=np.float32)
            if historical_world_from_lidar.shape != (4, 4):
                raise ValueError(
                    f"historical_world_from_lidar for camera '{cam_name}' must be a 4x4 matrix, got shape {historical_world_from_lidar.shape}")
            if historical_ego2img.shape != (4, 4):
                raise ValueError(
                    f"historical_ego2img for camera '{cam_name}' must be a 4x4 matrix, got shape {historical_ego2img.shape}")

            historical_lidar_from_current_lidar = (
                np.linalg.inv(historical_world_from_lidar) @ current_world_from_lidar
            )
            ego2img.append(historical_ego2img @ historical_lidar_from_current_lidar)

        return dict(filename=filenames, img_timestamp=timestamps, ego2img=ego2img)

    def __call__(self, results):
        curr_pose = results['world_from_lidar']
        curr_info = self.extract_info(results['cam_info'], curr_pose)

        prev_sweeps = results['cam_sweeps']['prev'][:self.prev_sweeps_num]
        next_sweeps = results['cam_sweeps']['next'][:self.next_sweeps_num]
        prev_sweeps = self.padding(prev_sweeps, self.prev_sweeps_num)
        next_sweeps = self.padding(next_sweeps, self.next_sweeps_num)

        prev_infos = [
            self.extract_info(sweep, curr_pose) if sweep is not None else curr_info
            for sweep in prev_sweeps
        ]
        next_infos = [
            self.extract_info(sweep, curr_pose) if sweep is not None else curr_info
            for sweep in next_sweeps
        ]

        # Temporal Order
        #   future (oldest -> nearest), current, past (nearest -> oldest)
        frame_infos = next_infos[::-1] + [curr_info] + prev_infos

        imgs, filenames, timestamps, ego2img = [], [], [], []
        for frame_info in frame_infos:
            filenames.extend(frame_info['filename'])
            timestamps.extend(frame_info['img_timestamp'])
            ego2img.extend(frame_info['ego2img'])
            for filename in frame_info['filename']:
                img = mmcv.imread(filename, self.color_type)
                if img is None:
                    raise FileNotFoundError(f"Image file not found: {filename}")
                imgs.append(img)

        expected_imgs = self.num_frames * self.num_views
        actual_lengths = {
            'imgs': len(imgs),
            'filenames': len(filenames),
            'timestamps': len(timestamps),
            'ego2img': len(ego2img)
        }
        invalid_lengths = {
            key: length
            for key, length in actual_lengths.items()
            if length != expected_imgs
        }
        if invalid_lengths:
            raise ValueError(
                f"Expected {expected_imgs} images, but got inconsistent lengths: {invalid_lengths}")

        results['img'] = imgs
        results['filename'] = filenames
        results['img_timestamp'] = timestamps
        results['ego2img'] = ego2img
        results['num_frames'] = self.num_frames
        results['num_views'] = self.num_views
        results['img_shape'] = [img.shape for img in imgs]
        results['ori_shape'] = [img.shape for img in imgs]
        results['pad_shape'] = [img.shape for img in imgs]
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label
