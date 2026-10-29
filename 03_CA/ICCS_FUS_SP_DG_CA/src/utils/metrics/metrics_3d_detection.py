import torch
from torchmetrics import Metric
from src.utils.centerpoint_utils import get_decoded_bbox_from_pred, nms_boxes
from mmcv.ops import box_iou_rotated
import logging
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from src.utils.centerpoint_utils import get_decoded_bbox_from_pred


logging.getLogger(__file__)

class Detection3DMetrics(Metric):
    # Evaluation uses discrete matching, so this pmetric is not differentiable.
    is_differentiable = False

    # A higher AP value indicates better detection performance.
    higher_is_better = True

    # Each update can process its batch independently of previous batches.
    full_state_update = False

    def __init__(
        self,
        x_min,
        y_min,
        metres_per_cell,
        score_threshold,
        thresh_iou,
        num_classes=10, 
    ):
        # Initialize TorchMetrics' state-management machinery.
        super().__init__()

        # BEV origin in metres, used to decode grid coordinates.
        self.x_min = x_min
        self.y_min = y_min

        # Physical size of one cell in the final prediction grid.
        self.metres_per_cell = metres_per_cell

        # Discard decoded predictions below this confidence.
        self.score_threshold = score_threshold

        # Minimum 3D IoU required for a prediction to match a GT object.
        self.thresh_iou = thresh_iou

        # Class IDs are expected to range from 0 to num_classes - 1.
        self.num_classes = num_classes

        # Store one row per prediction:
        # [confidence_score, true_positive_flag, class_id].
        #
        # true_positive_flag = 1 for a successful match, otherwise 0.
        # "cat" concatenates records across processes during compute().
        self.add_state(
            "records",
            default=[],
            dist_reduce_fx="cat",
        )

        # Count all GT objects of each class across the validation epoch.
        # Shape: [num_classes].
        # "sum" adds counts across processes during compute().
        self.add_state(
            "num_gt",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @torch.no_grad()  # Metrics do not need gradient tracking.
    def update(self, predictions, targets):
        """
        Process one batch and accumulate data needed for epoch AP.

        predictions:
            Raw CenterPoint outputs accepted by your decoder.

        targets:
            Batch dictionary containing targets["gt_boxes"].
            gt_boxes is a list of samples, each containing object dictionaries.
        """
        
        # Decode dense prediction maps into metric boxes.
        # results is a list with one dictionary per sample:
        # {
        #     "boxes":  [N, 7],
        #     "scores": [N],
        #     "labels": [N],
        # }^^^
        results = get_decoded_bbox_from_pred(
            predictions=predictions,
            score_threshold=self.score_threshold,
            x_min=self.x_min,
            y_min=self.y_min,
            metres_per_cell=self.metres_per_cell,
        )

        # IMPORTANT:
                # Rotated BEV NMS, independently per sample and class.
        # Separate from self.thresh_iou, which controls GT matching.
        results_postprocessed = nms_boxes(results=results)
        # Without NMS, duplicate predictions are evaluated as extra detections
        # and can become false positives.

        # Convert dataset object dictionaries into boxes and class tensors. (get metric tensors) (SAME LIKE PREDICTIONS)
        # {"boxes": ... , "labels": , ...}
        gt_batch = self.get_gt_stacks_from_ds(targets)

        # Prevent zip() from silently dropping unmatched batch entries.
        if len(results_postprocessed) != len(gt_batch):
            raise ValueError("Prediction and GT batch sizes differ")

        # Use the device on which this metric's registered state lives.
        device = self.num_gt.device

        # Match predictions only against GT from the SAME sample.
        for pred, gt in zip(results_postprocessed, gt_batch):

            # Predicted metric boxes: [N, 7].
            # Column order: [x, y, z_center, length, width, height, yaw].
            pboxes = pred["boxes"].detach().to(device).float()

            # Confidence and class ID for each predicted box: [N].
            pscores = pred["scores"].detach().to(device).float()
            plabels = pred["labels"].detach().to(device).long()

            # GT boxes [M, 7] and GT class IDs [M].
            gboxes = gt["boxes"]
            glabels = gt["labels"]

            # Evaluate each class separately.
            # A predicted car cannot match a GT pedestrian.
            for c in range(self.num_classes):

                # Boolean masks identifying objects of class c.
                pred_mask = plabels == c
                gt_mask = glabels == c

                # Keep only class-c predictions and GT.
                boxes = pboxes[pred_mask]       # [Nc, 7]
                scores = pscores[pred_mask]     # [Nc] (keep only scores that are pred_mask ,the others are discarded)
                gt_boxes = gboxes[gt_mask]      # [Mc, 7]

                # Count every GT object, including those never detected.
                # This count becomes the denominator for recall.
                self.num_gt[c] += len(gt_boxes)

                # Highest-confidence predictions get the first chance
                # to match a GT object.
                order = scores.argsort(descending=True)

                # Apply the SAME ordering to boxes and their scores.
                boxes = boxes[order]
                scores = scores[order]

                # Initially every prediction is a false positive.
                # Set an entry to 1 only after a successful GT match.
                tp = torch.zeros_like(scores)  # [Nc]

                # Track which GT objects have already been matched.
                # Each GT may contribute at most one true positive.
                matched_gt = torch.zeros(
                    len(gt_boxes),
                    dtype=torch.bool,
                    device=device,
                )  # [Mc]

                # Compare every prediction against every GT of this class.
                # ious[i, j] = 3D IoU of prediction i and GT j.
                ious = self.pairwise_iou(   
                    boxes,
                    gt_boxes,
                    mode="3d",
                )  # [Nc, Mc]

                # Walk through predictions in descending confidence order.
                for i in range(len(boxes)):

                    # With no GT of this class, all predictions remain FP.
                    if len(gt_boxes) == 0:
                        break

                    # Prevent reuse of GT objects that already matched.
                    # IoU is nonnegative, so -1 makes these ineligible.
                    available_ious = ious[i].masked_fill(
                        matched_gt, -1.0
                    )  # [Mc]

                    # Find the best remaining GT match for this prediction.
                    best_iou, gt_index = available_ious.max(dim=0)

                    # Accept the match only if its overlap is sufficient.
                    if best_iou >= self.thresh_iou:
                        tp[i] = 1.0
                        matched_gt[gt_index] = True

                # Store prediction results for epoch-level ranking.
                # Example row: [0.92, 1.0, 2.0]
                # means confidence 0.92, true positive, class 2.
                class_records = torch.stack(
                    [
                        scores,
                        tp,
                        torch.full_like(scores, float(c)),
                    ],
                    dim=1,
                )  # [Nc, 3]

                # An empty [0, 3] tensor is valid if this class had no predictions.
                self.records.append(class_records)  #holds records for all objects for all classes (scores,tp) independently


    def compute(self):
        """
        Compute AP from all batches accumulated since the last reset.
        """
        #Combine prediction records from all processed frames.
        #Shape: [total_number_of_predictions, 3].
        records = (
            dim_zero_cat(self.records)
            if len(self.records) > 0
            else torch.empty((0, 3), device=self.num_gt.device)
        )

        # Output dictionary and list of APs included in the class mean.
        metrics = {}
        metrics_2 = {}
        class_aps = []

        class_precisions=[]
        class_recalls=[]
        class_cum_tp = []
        class_cum_fp= []

        for c in range(self.num_classes):

            # AP is undefined when the evaluation set has no GT of this class.
            # Report NaN and exclude this class from the mAP calculation.
            if self.num_gt[c] == 0:
                metrics[f"AP_class_{c}"] = records.new_tensor(
                    float("nan")
                )
                continue

            # Select all predictions of class c across ALL frames.
            rows = records[records[:, 2] == c]

            # Rank those predictions globally by confidence.
            # Per-frame matching has already been completed in update().
            rows = rows[rows[:, 0].argsort(descending=True)]

            # Running TP and FP counts as we include lower-scoring predictions.
            # Example TP flags: [1, 0, 1] -> cumulative TP: [1, 1, 2].
            cumulative_tp = rows[:, 1].cumsum(dim=0)
            metrics[f"TP_class_{c}"] = (
                cumulative_tp[-1]
                if cumulative_tp.numel() > 0
                else self.num_gt.new_tensor(0).float()
            )
            cumulative_fp = (1.0 - rows[:, 1]).cumsum(dim=0)

            # Recall: fraction of all GT objects recovered so far.
            recall = cumulative_tp / self.num_gt[c]

            # Precision: fraction of included predictions that are correct.
            precision = cumulative_tp / (
                cumulative_tp + cumulative_fp
            ).clamp_min(1)

            # Add recall endpoints so we can integrate the full [0, 1] range.
            recall = torch.cat([
                recall.new_zeros(1),
                recall,
                recall.new_ones(1),
            ])

            # Precision beyond the achieved recall is zero.
            precision = torch.cat([
                precision.new_zeros(1),
                precision,
                precision.new_zeros(1),
            ])

            # Construct the interpolated precision envelope:
            # at each recall, use the highest precision at that recall or higher.
            # Reverse -> cumulative maximum -> reverse back.
            precision = torch.cummax(
                precision.flip(0),
                dim=0,
            ).values.flip(0)

            # Area under the interpolated precision–recall curve.
            # Each term is: recall interval width × precision for that interval.
            ap = (
                (recall[1:] - recall[:-1]) * precision[1:]
            ).sum()

            # Save this class's AP and include it in the class average.
            metrics[f"AP_class_{c}"] = ap
            class_aps.append(ap)

            # class_precisions.append(precision)
            # class_recalls.append(recall)
            # class_cum_tp.append(cumulative_tp)
            # class_cum_fp.append(cumulative_fp)
            # metrics_2[f"recalls_class_{c}"] = class_recalls[-1]  
            # metrics_2[f"cum_tp_class_{c}"] = class_cum_tp[-1]
            # metrics_2[f"cum_fp_class_{c}"] = class_cum_tp[-1]

        # mAP here means the mean across classes at ONE IoU threshold.
        metrics["mAP"] = (
            torch.stack(class_aps).mean()
            if class_aps
            else records.new_tensor(float("nan"))
        )
        # metrics["precision"] = precision

        return metrics,metrics_2

    def get_gt_stacks_from_ds(self, boxes):
        """
        Convert your dataset annotations into metric tensors.

        Assumptions:
        - boxes is a list of samples, each a list of object dictionaries.
        - obj["height"] contains z-center, NOT physical box height.
        - obj["type"] contains a numeric class ID.
        """

        batch = []
        device = self.num_gt.device

        # Process one sample at a time because object counts can differ.
        for sample in boxes:
            rows = []
            labels = []

            for obj in sample:
                dims = obj["3d_dimensions"]

                # Match the decoded prediction format exactly.
                rows.append([
                    obj["bbox"][0],    # Center x in metres.
                    obj["bbox"][1],    # Center y in metres.
                    obj["height"],    # Center z in metres.
                    dims["l"],        # Length.
                    dims["w"],        # Width.
                    dims["h"],        # Physical height.
                    obj["rotation"],  # Yaw in radians.
                ])

                # Class IDs must match the heatmap channel IDs.
                labels.append(int(obj["type"]))

            batch.append({
                # reshape preserves [0, 7] for samples with no GT objects.
                "boxes": torch.tensor(
                    rows,
                    dtype=torch.float32,
                    device=device,
                ).reshape(-1, 7),

                # One class label per GT box.
                "labels": torch.tensor(
                    labels,
                    dtype=torch.long,
                    device=device,
                ),
            })

        return batch

    def pairwise_iou(self, boxes1, boxes2, mode="3d"):
        
        """Boxes: [N, 7], [x, y, z_center, l, w, h, yaw].
        
        retuns volume io
        """
        if mode not in ("bev", "3d"):
            raise ValueError("mode must be 'bev' or '3d'")

        a = boxes1.detach().float()
        b = boxes2.detach().to(device=a.device, dtype=torch.float32)

        if len(a) == 0 or len(b) == 0:
            return a.new_zeros((len(a), len(b)))

        # MMCV's 'width' is the local-x extent: our length.
        # Its 'height' here is the local-y extent: our width.
        bev_a = a[:, [0, 1, 3, 4, 6]].contiguous()
        bev_b = b[:, [0, 1, 3, 4, 6]].contiguous()

        bev_iou = box_iou_rotated(bev_a, bev_b)

        if mode == "bev":
            return bev_iou

        area_a = a[:, 3] * a[:, 4]
        area_b = b[:, 3] * b[:, 4]

        # Recover intersection area from IoU = I / (A + B - I).
        intersection_area = (
            bev_iou * (area_a[:, None] + area_b[None, :])
            / (1 + bev_iou)
        )

        bottom_a = a[:, 2] - a[:, 5] / 2
        top_a = a[:, 2] + a[:, 5] / 2
        bottom_b = b[:, 2] - b[:, 5] / 2
        top_b = b[:, 2] + b[:, 5] / 2

        overlap_height = (
            torch.minimum(top_a[:, None], top_b[None, :])
            - torch.maximum(bottom_a[:, None], bottom_b[None, :])
        ).clamp_min(0)

        intersection_volume = intersection_area * overlap_height
        volume_a = area_a * a[:, 5]
        volume_b = area_b * b[:, 5]

        union_volume = (
            volume_a[:, None] + volume_b[None, :] - intersection_volume
        )

        return intersection_volume / union_volume.clamp_min(1e-8)


class custom_Detection3DMetrics(Metric):
    def __init__(self, x_min,
                        y_min,
                        metres_per_cell,
                        score_threshold,
                        thresh_iou):
        """
        score_threshold: Threshold for assigning a heatmap results to a keypoint
        thresh_iou: iou threshold
        """
        super().__init__()
        self.x_min = x_min
        self.y_min = y_min 
        self.metres_per_cell
        self.metres_per_cell = metres_per_cell
        self.score_threshold = score_threshold
        # self.add_state(...) for accumulated evaluation data

        self.iou_epoch = []

    def update(self, predictions, targets):
        """
        inputs:
            predictions: Predictions outputs of centerpoint (bev)
            targets: Targets in REAL
            Here, z is the box-center elevation, yaw is in radians, and positive yaw rotates local +x toward +y. 
            If your GT uses bottom elevation, convert it first: z_center = z_bottom + height / 2.
        """
        # Process each frame: matching, scores, TP/FP, GT counts.
        results = get_decoded_bbox_from_pred(predictions=predictions, score_threshold=self.score_threshold)  #get REAL DECODED results in stack format
        targets = self.get_gt_stacks_from_ds(boxes = targets["gt_boxes"])

        iou_list=[] #batch
        for b in results.shape[0]:
            iou_list.append(self.pairwise_iou(boxes1= results[b]["boxes"] , 
                                              boxes2=targets[b]["boxes"],
                                              mode="3d"))
            
        self.iou_epoch.append(iou_list) #epoch 
        

    # def compute(self):
    #     # Compute metrics over the entire accumulated dataset.
    #     ...

    def get_gt_stacks_from_ds(self,boxes):
        """
        boxes batches , frrom dataset creation stage
        """
        boxes_batch=[]
        for b in boxes.shape[0]:
            boxes_batch.append(boxes[b]["bbox"][0],boxes[b]["bbox"][1], 
                               boxes[b]["height"] , 
                               boxes[b]["3d_dimensions"]["l"], 
                               boxes[b]["3d_dimensions"]["h"], 
                               boxes[b]["3d_dimensions"]["w"], 
                               boxes[b]["rotation"])
        


    @torch.no_grad()
    def pairwise_iou(self, boxes1, boxes2, mode="3d"):



        """Boxes: [N, 7], [x, y, z_center, l, w, h, yaw].
        
        retuns volume io
        """
        if mode not in ("bev", "3d"):
            raise ValueError("mode must be 'bev' or '3d'")

        a = boxes1.detach().float()
        b = boxes2.detach().to(device=a.device, dtype=torch.float32)

        if len(a) == 0 or len(b) == 0:
            return a.new_zeros((len(a), len(b)))

        # MMCV's 'width' is the local-x extent: our length.
        # Its 'height' here is the local-y extent: our width.
        bev_a = a[:, [0, 1, 3, 4, 6]].contiguous()
        bev_b = b[:, [0, 1, 3, 4, 6]].contiguous()

        bev_iou = box_iou_rotated(bev_a, bev_b)

        if mode == "bev":
            return bev_iou

        area_a = a[:, 3] * a[:, 4]
        area_b = b[:, 3] * b[:, 4]

        # Recover intersection area from IoU = I / (A + B - I).
        intersection_area = (
            bev_iou * (area_a[:, None] + area_b[None, :])
            / (1 + bev_iou)
        )

        bottom_a = a[:, 2] - a[:, 5] / 2
        top_a = a[:, 2] + a[:, 5] / 2
        bottom_b = b[:, 2] - b[:, 5] / 2
        top_b = b[:, 2] + b[:, 5] / 2

        overlap_height = (
            torch.minimum(top_a[:, None], top_b[None, :])
            - torch.maximum(bottom_a[:, None], bottom_b[None, :])
        ).clamp_min(0)

        intersection_volume = intersection_area * overlap_height
        volume_a = area_a * a[:, 5]
        volume_b = area_b * b[:, 5]

        union_volume = (
            volume_a[:, None] + volume_b[None, :] - intersection_volume
        )

        return intersection_volume / union_volume.clamp_min(1e-8)