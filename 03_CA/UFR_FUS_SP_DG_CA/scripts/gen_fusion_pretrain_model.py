import torch
import re

lidar_ckpt = torch.load('pretrain/dal-tiny-map66.9-nds71.1.pth')
img_ckpt = torch.load('pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth')

lidar_dict = lidar_ckpt['state_dict']
img_dict = img_ckpt['state_dict']

lidar_prefix_keys_list=['pts_backbone', 'pts_middle_encoder', 'pts_neck']
for key in list(lidar_dict.keys()):
    flag=False
    for prefix in lidar_prefix_keys_list:
        if key.startswith(prefix):
            flag=True
            break
    if not flag:
        del lidar_dict[key]
           
img_prefix_keys_list=['backbone']
for prefix in img_prefix_keys_list:
    for key in img_dict:
        if key.startswith(prefix):
            new_key=re.sub('backbone', 'img_backbone', key)
            lidar_dict[new_key] = img_dict[key]

torch.save({'state_dict': lidar_dict}, 'pretrain/fusion_pretrain_model.pth')
