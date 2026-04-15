'''
Author: George Panaretou

Dual-ego heterogeneous inference dataset.

Instead of assigning different modalities to different physical vehicles,
the ego vehicle is presented as TWO virtual agents:
  - Agent 0: ego with lidar (m1)
  - Agent 1: ego with camera (m2)  [only when use_cav >= 2]

GT boxes are still collected from all real vehicles in the scene.
The pairwise transformation matrix between the two virtual agents is identity
(same physical vehicle, same pose).

use_cav = 1: single-modality ego (lidar only)
use_cav = 2: dual-modality ego (lidar + camera)
'''

import copy
import math
from collections import OrderedDict
import numpy as np
import torch
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils import box_utils as box_utils
from opencood.utils.common_utils import merge_features_to_dict, compute_iou, convert_format
from opencood.utils.transformation_utils import get_pairwise_transformation
from opencood.utils.pcd_utils import mask_points_by_range
from opencood.data_utils.datasets.heter_infer.intermediate_heter_infer_fusion_dataset import getIntermediateheterinferFusionDataset


def getIntermediateheterdualegoinferFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    Intermediate_heter_Infer_Fusion_Dataset = getIntermediateheterinferFusionDataset(cls)

    class Intermediate_heter_Dual_Ego_Infer_Fusion_Dataset(Intermediate_heter_Infer_Fusion_Dataset):
        """
        Ego vehicle is presented as two virtual agents sharing the same pose:
          - Agent 0: lidar modality (m1)
          - Agent 1: camera modality (m2)

        GT boxes are collected from all real vehicles (unchanged from parent).
        Only the ego's sensors contribute to the collaborative feature extraction.
        """

        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            assert self.sensor_type_dict.get('m1') == 'lidar', \
                "m1 must be lidar for dual-ego mode (check modality_setting in hypes)"
            assert self.sensor_type_dict.get('m2') == 'camera', \
                "m2 must be camera for dual-ego mode (check modality_setting in hypes)"
            print("\nDual-Ego Mode: Agent 0 = ego lidar (m1), Agent 1 = ego camera (m2)")
            print(f"use_cav={self.use_cav}: {'lidar+camera' if self.use_cav >= 2 else 'lidar only'}\n")

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
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
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
            # Virtual dual-ego agent count: at most 2 (lidar + camera)
            # ---------------------------------------------------------------
            cav_num = min(self.use_cav, 2)

            # ---------------------------------------------------------------
            # pairwise_t_matrix: build a fake dict where ego appears cav_num
            # times (same pose => identity transformation between agents).
            # ---------------------------------------------------------------
            fake_data_dict = OrderedDict()
            fake_data_dict[ego_id] = base_data_dict[ego_id]          # agent 0: ego lidar
            if cav_num >= 2:
                fake_data_dict['ego_cam'] = base_data_dict[ego_id]   # agent 1: ego camera (same pose)

            pairwise_t_matrix = get_pairwise_transformation(
                fake_data_dict, self.max_cav, self.proj_first
            )

            # ---------------------------------------------------------------
            # lidar_poses: ego pose repeated cav_num times (both agents share
            # the same physical location).
            # ---------------------------------------------------------------
            ego_pose = base_data_dict[ego_id]['params']['lidar_pose']
            ego_pose_clean = base_data_dict[ego_id]['params']['lidar_pose_clean']
            lidar_poses = np.array([ego_pose] * cav_num).reshape(-1, 6)
            lidar_poses_clean = np.array([ego_pose_clean] * cav_num).reshape(-1, 6)

            # ---------------------------------------------------------------
            # GT box collection from ALL real agents in cav_id_list.
            # Each real vehicle contributes its detected objects to the scene GT.
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
            # Feature extraction: ego as m1 (lidar) — always Agent 0
            # ---------------------------------------------------------------
            self.generate_object_center = self.generate_object_center_lidar
            ego_as_m1 = {**ego_cav_base, 'modality_name': 'm1'}
            processed_m1 = self.get_item_single_car(ego_as_m1, ego_cav_base)

            input_list_m1.append(processed_m1['processed_features_m1'])
            agent_modality_list.append('m1')

            if self.visualize or self.kd_flag:
                projected_lidar_stack.append(processed_m1['projected_lidar'])

            if self.supervise_single or self.heterogeneous:
                single_label_list.append(processed_m1['single_label_dict'])
                single_object_bbx_center_list.append(processed_m1['single_object_bbx_center'])
                single_object_bbx_mask_list.append(processed_m1['single_object_bbx_mask'])

            # ---------------------------------------------------------------
            # Feature extraction: ego as m2 (camera) — Agent 1 (if use_cav>=2)
            # ---------------------------------------------------------------
            if cav_num >= 2:
                self.generate_object_center = self.generate_object_center_camera
                ego_as_m2 = {**ego_cav_base, 'modality_name': 'm2'}
                processed_m2 = self.get_item_single_car(ego_as_m2, ego_cav_base)

                input_list_m2.append(processed_m2['image_inputs_m2'])
                agent_modality_list.append('m2')

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(processed_m2['projected_lidar'])

                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(processed_m2['single_label_dict'])
                    single_object_bbx_center_list.append(processed_m2['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(processed_m2['single_object_bbx_mask'])

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
