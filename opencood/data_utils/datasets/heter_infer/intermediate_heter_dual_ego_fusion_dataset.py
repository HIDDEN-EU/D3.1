'''
Author: George Panaretou

Dual-ego heterogeneous inference dataset.

Instead of assigning different modalities to different physical vehicles,
each physical car is presented as TWO virtual agents:
  - Agent (2i):   car_i with its lidar modality  (derived from sensor_type_dict)
  - Agent (2i+1): car_i with its camera modality (derived from sensor_type_dict)

GT boxes are still collected from all real vehicles in the scene.
The pairwise transformation between the two virtual agents of the same car is
identity (same physical vehicle, same pose); between cars it is the real transform.

use_cav controls the number of *physical cars* that contribute sensors:
  use_cav = 1  →  2 virtual agents  (ego lidar + ego camera)
  use_cav = 2  →  4 virtual agents  (ego lidar/camera + car2 lidar/camera)
  use_cav = N  →  2N virtual agents (N cars × lidar + camera)
  max use_cav  =  4

NOTE: max_cav in the hypes yaml must be >= 2*use_cav (e.g. 8 for use_cav=4),
otherwise pairwise_t_matrix will silently truncate agents.
'''

import math
from collections import OrderedDict

import numpy as np
import torch

from opencood.data_utils.datasets.heter_infer.intermediate_heter_infer_fusion_dataset import (
    getIntermediateheterinferFusionDataset,
)
from opencood.utils import box_utils as box_utils
from opencood.utils.common_utils import (
    compute_iou,
    convert_format,
    merge_features_to_dict,
)
from opencood.utils.pcd_utils import mask_points_by_range
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.transformation_utils import get_pairwise_transformation


def getIntermediateheterdualegoinferFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    Intermediate_heter_Infer_Fusion_Dataset = getIntermediateheterinferFusionDataset(cls)

    class Intermediate_heter_Dual_Ego_Infer_Fusion_Dataset(Intermediate_heter_Infer_Fusion_Dataset):
        """
        Each physical car contributes two virtual agents (lidar + camera).
        GT boxes are collected from all real vehicles (unchanged from parent).
        Modality names are derived from sensor_type_dict — not hardcoded.
        """

        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # Derive lidar/camera modality names from sensor_type_dict
            self._lidar_mod  = next(m for m, t in self.sensor_type_dict.items() if t == 'lidar')
            self._camera_mod = next(m for m, t in self.sensor_type_dict.items() if t == 'camera')
            assert self._lidar_mod is not None, "No lidar modality found in sensor_type_dict"
            assert self._camera_mod is not None, "No camera modality found in sensor_type_dict"
            print(f"\nDual-Ego Mode: lidar_mod={self._lidar_mod}, camera_mod={self._camera_mod}")
            print(f"use_cav={self.use_cav}: {self.use_cav} physical car(s), {self.use_cav*2} virtual agents\n")

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # Find the ego vehicle (always the first entry)
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break

            assert cav_id == list(base_data_dict.keys())[0], \
                "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            lidar_mod  = self._lidar_mod
            camera_mod = self._camera_mod

            input_list_m1 = []
            input_list_m2 = []
            input_list_m3 = []
            input_list_m4 = []

            agent_modality_list = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            exclude_agent = []
            cav_id_list = []

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # ---------------------------------------------------------------
            # Build cav_id_list: all real agents within communication range.
            # These are used for GT box collection only.
            # ---------------------------------------------------------------
            for cav_id, selected_cav_base in base_data_dict.items():
                distance = math.sqrt(
                    (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                    (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
                )
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue
                cav_id_list.append(cav_id)

            if len(cav_id_list) == 0:
                return None

            for cav_id in exclude_agent:
                base_data_dict.pop(cav_id)

            # ---------------------------------------------------------------
            # Box alignment (if enabled) — same logic as parent
            # ---------------------------------------------------------------
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import (
                    box_alignment_relative_sample_np,
                )
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list']
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose']
                                      for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent)
                                              for cur_agent in cur_agent_id_list]

                    pred_corners_list = [np.array(all_agent_corners_list[i], dtype=np.float64)
                                         for i in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[i], dtype=np.float64)
                                        for i in cur_agent_in_all_agent]

                    if sum([len(p) for p in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(
                            pred_corners_list, cur_agnet_pose,
                            uncertainty_list=uncertainty_list, **self.box_align_args
                        )
                        cur_agnet_pose[:, [0, 1, 4]] = refined_pose
                        for i, cav_id in enumerate(cav_id_list):
                            base_data_dict[cav_id]['params']['lidar_pose'] = \
                                cur_agnet_pose[i].tolist()

                    # Refresh ego references after potential pose refinement
                    ego_cav_base = base_data_dict[ego_id]
                    ego_lidar_pose = ego_cav_base['params']['lidar_pose']

            # ---------------------------------------------------------------
            # Select physical cars to contribute sensors: up to use_cav cars
            # in cav_id_list order (ego is always first).
            # ---------------------------------------------------------------
            physical_cars = []
            for cid in cav_id_list:
                physical_cars.append(cid)
                if len(physical_cars) >= self.use_cav:
                    break

            cav_num = len(physical_cars) * 2  # two virtual agents per physical car

            # ---------------------------------------------------------------
            # pairwise_t_matrix: each physical car appears twice with same pose
            # (identity between its two virtual agents; real transform between cars).
            # ---------------------------------------------------------------
            fake_data_dict = OrderedDict()
            for cid in physical_cars:
                fake_data_dict[cid]          = base_data_dict[cid]   # lidar virtual agent
                fake_data_dict[f'{cid}_cam'] = base_data_dict[cid]   # camera virtual agent

            
            pairwise_t_matrix = get_pairwise_transformation(
                fake_data_dict, self.max_cav, self.proj_first
            )

            # ---------------------------------------------------------------
            # lidar_poses: each physical car's pose repeated twice.
            # ---------------------------------------------------------------
            lidar_poses_list = []
            lidar_poses_clean_list = []
            for cid in physical_cars:
                pose       = base_data_dict[cid]['params']['lidar_pose']
                pose_clean = base_data_dict[cid]['params']['lidar_pose_clean']
                lidar_poses_list       += [pose, pose]
                lidar_poses_clean_list += [pose_clean, pose_clean]
            lidar_poses       = np.array(lidar_poses_list).reshape(-1, 6)
            lidar_poses_clean = np.array(lidar_poses_clean_list).reshape(-1, 6)

            # ---------------------------------------------------------------
            # GT box collection from ALL real agents in cav_id_list.
            # ---------------------------------------------------------------
            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                sensor_type = self.sensor_type_dict[selected_cav_base['modality_name']]

                if not self.visualize:
                    self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")
                else:
                    self.generate_object_center = self.generate_object_center_lidar

                object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                    [selected_cav_base], ego_cav_base['params']['lidar_pose_clean']
                )
                object_stack.append(object_bbx_center[object_bbx_mask == 1])
                object_id_stack += object_ids

            # ---------------------------------------------------------------
            # Feature extraction: for each physical car, process as lidar then camera.
            # Follows the same eval(f"input_list_{modality_name}") pattern as parent.
            # ---------------------------------------------------------------
            for cid in physical_cars:
                cav_base = base_data_dict[cid]

                # --- lidar virtual agent ---
                self.generate_object_center = self.generate_object_center_lidar
                cav_as_lidar = {**cav_base, 'modality_name': lidar_mod}

                if cid == ego_id and self.params.get('lidar_corrupt', {}).get('enabled', False):
                    ratio = self.params['lidar_corrupt']['dropout_ratio']
                    lidar = cav_as_lidar['lidar_np']
                    keep = np.random.rand(len(lidar)) > ratio
                    cav_as_lidar['lidar_np'] = lidar[keep]

                processed_lidar = self.get_item_single_car(cav_as_lidar, ego_cav_base)
                eval(f"input_list_{lidar_mod}").append(
                    processed_lidar[f"processed_features_{lidar_mod}"]
                )
                agent_modality_list.append(lidar_mod)

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(processed_lidar['projected_lidar'])

                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(processed_lidar['single_label_dict'])
                    single_object_bbx_center_list.append(processed_lidar['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(processed_lidar['single_object_bbx_mask'])

                # --- camera virtual agent ---
                self.generate_object_center = self.generate_object_center_camera

                # Gpanaretou: force agent sensor modality to m4 since m2 is broken (m2 and m4 are both camera)
                if camera_mod == 'm2':
                    camera_mod = 'm4'

                cav_as_camera = {**cav_base, 'modality_name': camera_mod}
                processed_camera = self.get_item_single_car(cav_as_camera, ego_cav_base)
                eval(f"input_list_{camera_mod}").append(
                    processed_camera[f"image_inputs_{camera_mod}"]
                )
                agent_modality_list.append(camera_mod)

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(processed_camera['projected_lidar'])

                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(processed_camera['single_label_dict'])
                    single_object_bbx_center_list.append(processed_camera['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(processed_camera['single_object_bbx_mask'])

            # ---------------------------------------------------------------
            # Single-view supervision labels
            # ---------------------------------------------------------------
            if self.supervise_single or self.heterogeneous:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                })

            if self.kd_flag:
                stack_lidar_np = np.vstack(projected_lidar_stack)
                stack_lidar_np = mask_points_by_range(
                    stack_lidar_np, self.params['preprocess']['cav_lidar_range']
                )
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({'teacher_processed_lidar': stack_feature_processed})

            # ---------------------------------------------------------------
            # Merge GT boxes from all real agents (deduplicate by object ID)
            # ---------------------------------------------------------------
            if self.params['fusion']['dataset'] == 'dairv2x':
                if len(object_stack) == 1:
                    object_stack = object_stack[0]
                else:
                    ego_boxes_np = object_stack[0]
                    cav_boxes_np = object_stack[1]
                    order = self.params['postprocess']['order']
                    ego_corners_np = box_utils.boxes_to_corners_3d(ego_boxes_np, order)
                    cav_corners_np = box_utils.boxes_to_corners_3d(cav_boxes_np, order)
                    ego_polygon_list = list(convert_format(ego_corners_np))
                    cav_polygon_list = list(convert_format(cav_corners_np))
                    iou_thresh = 0.05

                    gt_boxes_from_cav = []
                    for i in range(len(cav_polygon_list)):
                        cav_polygon = cav_polygon_list[i]
                        ious = compute_iou(cav_polygon, ego_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_boxes_from_cav.append(cav_boxes_np[i])

                    if len(gt_boxes_from_cav):
                        object_stack_from_cav = np.stack(gt_boxes_from_cav)
                        object_stack = np.vstack([ego_boxes_np, object_stack_from_cav])
                    else:
                        object_stack = ego_boxes_np

                unique_indices = np.arange(object_stack.shape[0])
                object_id_stack = np.arange(object_stack.shape[0])
            else:
                unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # ---------------------------------------------------------------
            # Merge sensor inputs per modality
            # ---------------------------------------------------------------
            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "lidar":
                    merged_feature_dict = merge_features_to_dict(
                        eval(f"input_list_{modality_name}")
                    )
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_feature_dict})
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(
                        eval(f"input_list_{modality_name}"), merge='stack'
                    )
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_image_inputs_dict})

            processed_data_dict['ego'].update({'agent_modality_list': agent_modality_list})

            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=self.anchor_box,
                mask=mask
            )

            processed_data_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses,
            })

            if self.visualize:
                processed_data_dict['ego'].update({
                    'origin_lidar': np.vstack(projected_lidar_stack)
                })

            processed_data_dict['ego'].update({
                'sample_idx': idx,
                'cav_id_list': cav_id_list,
            })

            return processed_data_dict

    return Intermediate_heter_Dual_Ego_Infer_Fusion_Dataset
