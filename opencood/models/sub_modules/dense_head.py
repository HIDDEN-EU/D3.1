# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from torch import nn


class Head(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.conv_box = nn.Conv2d(args['num_input'], args['num_pred'], 1)  # 128 -> 14
        self.conv_cls = nn.Conv2d(args['num_input'], args['num_cls'], 1)   # 128 -> 2
        self.conv_dir = nn.Conv2d(args['num_input'], args['num_dir'], 1)  # 128 -> 4
        self.conv_iou = nn.Conv2d(args['num_input'], args['num_dir'], 1, bias=False)

    def forward(self, x):
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        dir_preds = self.conv_dir(x)  # dir_preds.shape=[8, w, h, 4]
        iou_preds = self.conv_iou(x)

        ret_dict = {"reg_preds": box_preds, \
                    "cls_preds": cls_preds, \
                    "dir_preds": dir_preds, \
                    "iou_preds": iou_preds}

        return ret_dict