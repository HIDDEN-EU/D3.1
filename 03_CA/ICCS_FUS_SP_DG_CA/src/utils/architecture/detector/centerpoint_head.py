
    
  
"""Standalone standard (non-DCN) CenterPoint detection head, PyTorch only.

Adapted from tianweiy/CenterPoint, master, retrieved 2026-10-07:
https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/bbox_heads/center_head.py

CenterHead preserves the original forward return: (list of task dicts, features).
CenterPointHead provides a single-task dict and renames 'hm' to 'heatmap'.
No losses, target generation, decoding, NMS, or pretrained weights are included.
All predictions are raw: no sigmoid, exp, or angle conversion in forward.

Example:
    head = CenterPointHead(in_channels=128, num_classes=3)
    predictions = head(bev_features)  # [B, 128, H, W] -> dict of dense maps

Original structure and branch initialization are retained for the standard
64-channel, two-convolution configuration. Channel tracking also supports other
widths and convolution counts. Original standard-head state_dict paths are
retained in CenterHead (shared_conv, tasks.N.BRANCH.*).

MIT License
Copyright (c) 2020-2021 Tianwei Yin and Xingyi Zhou
Portions from Det3D: Copyright (c) 2019 朱本金

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from torch import nn


class SepHead(nn.Module):
    """Independent dense prediction branches; heads[name]=(channels, convs)."""

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=3,
                 bn=True, init_bias=-2.19):
        super().__init__()
        if final_kernel < 1 or final_kernel % 2 != 1:
            raise ValueError('final_kernel must be a positive odd integer')
        self.heads = dict(heads)
        for name, (out_channels, num_conv) in self.heads.items():
            if out_channels < 1 or num_conv < 1:
                raise ValueError('Branch channels and convolution counts must be positive')
            layers = []
            current_channels = in_channels
            for _ in range(num_conv - 1):
                layers.append(nn.Conv2d(current_channels, head_conv, final_kernel,
                                        padding=final_kernel // 2, bias=True))
                if bn:
                    layers.append(nn.BatchNorm2d(head_conv))
                layers.append(nn.ReLU())
                current_channels = head_conv
            layers.append(nn.Conv2d(current_channels, out_channels, final_kernel,
                                    padding=final_kernel // 2, bias=True))
            branch = nn.Sequential(*layers)
            if name == 'hm':
                nn.init.constant_(branch[-1].bias, init_bias)
            else:
                for module in branch.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                                nonlinearity='relu')
                        nn.init.zeros_(module.bias)
            self.add_module(name, branch)

    def forward(self, x):
        return {name: getattr(self, name)(x) for name in self.heads}


class CenterHead(nn.Module):
    """Original-style task grouping and forward interface; no det3d imports."""

    def __init__(self, 
                 in_channels, 
                 tasks, 
                 common_heads=None,
                 share_conv_channel=64, 
                 head_conv=64, 
                 num_hm_conv=2,
                 init_bias=-2.19):
        super().__init__()
        if not tasks or any(not task['class_names'] for task in tasks):
            raise ValueError('Provide at least one task with nonempty class_names')
        if common_heads is None:
            common_heads = dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2))
        self.class_names = [list(task['class_names']) for task in tasks]
        self.num_classes = [len(names) for names in self.class_names]
        self.in_channels = in_channels
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel, 3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True),
        )
        self.tasks = nn.ModuleList()
        for num_classes in self.num_classes:
            branches = dict(common_heads)
            branches['hm'] = (num_classes, num_hm_conv)
            self.tasks.append(SepHead(share_conv_channel, branches,
                                      head_conv=head_conv, init_bias=init_bias))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError('Expected [B, {}, H, W], received {}'.format(
                self.in_channels, tuple(x.shape)))
        shared = self.shared_conv(x)
        return [task(shared) for task in self.tasks], shared


class CenterPointHead(CenterHead):
    """Single-task adapter returning heatmap/reg/height/dim/rot[/vel].

    num_classes excludes background. Regression maps are shared across classes.
    This changes output names/container only, not the numerical predictions.
    """

    def __init__(self, in_channels, num_classes, with_velocity=False,
                 share_conv_channel=64, head_conv=64):
        if num_classes < 1:
            raise ValueError('num_classes must be positive')
        heads = dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2))
        if with_velocity:
            heads['vel'] = (2, 2)
        tasks = [dict(class_names=[str(i) for i in range(num_classes)])]
        super().__init__(in_channels, tasks, common_heads=heads,
                         share_conv_channel=share_conv_channel, head_conv=head_conv)

    def forward(self, x):
        task_outputs, _ = super().forward(x)
        predictions = task_outputs[0]
        predictions['heatmap'] = predictions.pop('hm').sigmoid()
        return predictions

