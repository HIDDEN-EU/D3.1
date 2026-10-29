from math import floor
import sys
from pyparsing import Optional
import torch
from mmcv.ops import box_iou_rotated, nms_rotated
from typing import List, Tuple

def nms_boxes( results):
        nms_iou_threshold = 0.2

        for pred in results:
            boxes = pred["boxes"]    # [N, 7]: x, y, z, l, w, h, yaw
            scores = pred["scores"]  # [N]
            labels = pred["labels"]  # [N]

            if boxes.shape[0] == 0:
                continue

            kept_indices = []

            for class_id in labels.unique():
                class_indices = torch.where(labels == class_id)[0]

                # MMCV expects [center_x, center_y, local_x_size,
                #               local_y_size, yaw].
                bev_boxes = (
                    boxes[class_indices][:, [0, 1, 3, 4, 6]]
                    .detach()
                    .float()
                    .contiguous()
                )
                class_scores = (
                    scores[class_indices].detach().float().contiguous()
                )

                _, keep = nms_rotated(
                    bev_boxes,
                    class_scores,
                    nms_iou_threshold,
                )

                # Convert indices within this class to original indices.
                kept_indices.append(class_indices[keep.long()])

            keep = torch.cat(kept_indices)
            keep = keep[scores[keep].argsort(descending=True)]

            # Apply identical filtering to all three tensors.
            pred["boxes"] = boxes[keep]
            pred["scores"] = scores[keep]
            pred["labels"] = labels[keep]

        return results


def encode_gt_batches(targets_metric:List, 
                    BEV_RANGE_HORZ: int,
                    BEV_RANGE_VERT: int,
                    BEV_RANGE_HORZ_MIN:int,
                    BEV_RANGE_VERT_MIN:int, 
                    metres_per_cell:float,
                    NUM_CLASSES:int,
                    device:Optional  )->Tuple:
    """
    GT ENCODING SCHEME FOR CENTERPOINT DETECTION

    encode heatmap head: return gaussian targets on the radius around class. In the grid you get zeros everywhere apart from obj centers where maximum is assigned and gaussian decay around it at radius = r.

    encode offset head: return zeros everywhere apart from centre cx,cy 

    encode height,


    encode dim: logarithm

    encode rot: sine 

    INputs
        targets_metric: targets in metric coords [B,N] b batches N detections 
            batches of lists of gt dictionaries of loc,dim,rot
        BEV_RANGE_HORZ: horizontal bev_range max
        BEV_RANGE_VERRT; vert bev_range max 

    RETURN:
        target: heatmap targets 
    """

    #targets metric [B , N , dicts]
    #make bbox
    
    # for sample in targets_metric: #iterate over batch:
        

    boxes_batch = [[obj["bbox"] for obj in sample] for sample in targets_metric]
        
    classes_batch = [[obj["type"] for obj in sample] for sample in targets_metric]

    pred_shape = (len(targets_metric) , NUM_CLASSES , BEV_RANGE_VERT , BEV_RANGE_HORZ )

    x_min = BEV_RANGE_HORZ_MIN
    y_min = BEV_RANGE_VERT_MIN
    
    H = BEV_RANGE_VERT
    W = BEV_RANGE_HORZ
    ###==================encode targets =======================
    target_heatmap = make_heatmap_targets(boxes_batch=boxes_batch,
                                            classes_batch=classes_batch,
                                            pred_shape=pred_shape,
                                            x_min=x_min,
                                            y_min=y_min,
                                            metres_per_cell=metres_per_cell)



    target_offsets, \
    target_heights, \
    target_sizes, \
    targets_rots, \
    target_mask = make_offset_targets(bboxes_batch=boxes_batch,                                                                                                                 
                                    heights_batch = 
                                        [[obj["height"] for obj in sample] for sample in targets_metric],
                                        
                                    rots_batch = [[obj["rotation"] for obj in sample] for sample in targets_metric],

                                    size_batch = [[obj["3d_dimensions"] for obj in sample] for sample in targets_metric],

                                    x_min=x_min,
                                    y_min=y_min, 
                                    metres_per_cell=metres_per_cell,H=H,W=W
                                                        
                                                                                            )

    return {"target_heatmap":target_heatmap.to(device) , 
            "target_offsets":target_offsets.to(device), 
            "target_height":target_heights.to(device), 
            "target_size":target_sizes.to(device), 
            "target_rots":targets_rots.to(device), 
            "target_mask":target_mask.to(device)}


# def make_height_targets(heights_batch , H ,W ):
#     B = len(heights_batch)
#     targets = torch.zeros(B, 1 , H , W)
#     for b in range(B):
#         for heights in heights_batch(b):
            


def make_offset_targets(bboxes_batch, 
                        heights_batch,
                        rots_batch,
                        size_batch , 
                        x_min , 
                        y_min , 
                        metres_per_cell,
                        H,
                        W,):
    """
    make targets for offset head of centerpoint , height size and rotation heads 

    return the difference of actual prediction in cell vs the cell of ground truth , which can be different due to discretization error.

    """

    B = len(bboxes_batch) 
    target_offsets = torch.zeros(B ,2 , H , W)
    targets_heights= torch.zeros(B, 1, H , W)
    targets_size = torch.zeros(B , 3 , H , W)
    targets_rots = torch.zeros (B, 2 , H , W)
    target_mask = torch.zeros(B, 1,H , W)
    for b in range(B):
        for bbox,height,rots,size in zip(bboxes_batch[b],heights_batch[b],rots_batch[b],size_batch[b]):
            x,y=bbox[0],bbox[1]

            ux = (x - x_min) / metres_per_cell
            uy = (y - y_min) / metres_per_cell

            cx = int(floor(ux))
            cy = int(floor(uy))

            if not (0 <= cx < W and 0 <= cy < H):
                continue

            target_offsets[b , : , cy,cx] =  torch.stack([torch.tensor(ux - cx), torch.tensor(uy - cy)])
            targets_heights[b, 0, cy, cx] = height
            targets_size[b, :, cy, cx] = torch.log(torch.tensor([size["w"],size["l"],size["h"]]))
            targets_rots[b, :, cy, cx] = torch.stack([
                torch.sin(torch.tensor(rots)),
                torch.cos(torch.tensor(rots)),
            ])
            target_mask[b,: , cy,cx] = 1
            
            check_infinite_dims(dims=torch.tensor(
                [size["w"], size["l"], size["h"]],
                dtype=torch.float32,
            ) , b=b , bbox=bbox,size=size,obj_idx=3)

    
    return target_offsets,targets_heights,targets_size,targets_rots,target_mask 


def make_heatmap_targets(boxes_batch, 
                         classes_batch, 
                         pred_shape,
                         x_min, 
                         y_min, 
                         metres_per_cell):
    """
    Here we have a gaussian normal centered on the keypoint. We set the target at keypoint equal to the gaussian
    at the centre (maximal =1 ) and exp decay around the keyframe to get 2radius+1 rectanggle.
    The reason is purely better training convergence instead of abrupt change from 0 to 1 in non-keypoint,keypoint 
    BEV cells. 

    the patch is a 4d tensor centered around keypoint, with b batch |C| classes and H,W metric space. the class
    constitues an ascending discrete metric predefined. All targets are assigned ranges in the [0,1] range

    Then perform training of the heatmap head with the focal loss.

    x_min: min x-coord BEV
    y_min min y-coord BEV
    metres_per_cell: metric length of cell in BEV 
    
    """

    B, num_classes, H, W = pred_shape

    #construct target tensor , this is for heatmap head of centerpoint
    target = torch.zeros(pred_shape)

    radius = 2
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=target.device),
        torch.arange(-radius, radius + 1, device=target.device),
    )


    #gaussian around keybpoint
    gaussian = torch.exp(-(xx**2 + yy**2) / (2 * 1.0**2))

    for b in range(B):
        #iterate in given ground truth database 
        for box,  class_name in zip(boxes_batch[b], classes_batch[b]):
            # box centre is in vehicle LiDAR metres
            x, y = box[0], box[1]  
            #box center in BEV cell discrete coordinates
            cx = int(torch.floor(torch.tensor((x - x_min) / metres_per_cell)))
            cy = int(torch.floor(torch.tensor((y - y_min) / metres_per_cell)))
            class_name = int(class_name)

            if not (0 <= cx < W and 0 <= cy < H):
                continue

            #take the coordinates which lie di rectly inside keypoint of heatmap only
            x0, x1 = max(0, cx - radius), min(W, cx + radius + 1) #max and min in case the boxes lie outside of the 2d range
            y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)

            #target is initially zeros
            patch = target[b, class_name, y0:y1, x0:x1]


            g = gaussian[
                y0 - (cy - radius): y1 - (cy - radius),
                x0 - (cx - radius): x1 - (cx - radius),
            ]

            target[b, class_name, y0:y1, x0:x1] = torch.maximum(patch, g)

    return target

def check_infinite_dims(dims,b,obj_idx,bbox,size):
            # These are the actual object's dimensions before log encoding.
            if not torch.isfinite(dims).all() or (dims <= 0).any():
                raise ValueError(
                    f"Invalid dimensions: batch={b}, object={obj_idx}, "
                    f"bbox={bbox}, dimensions={size}"
                )


def decode_predictions(predictions , score_threshold):
    """
    decode predictions of type centerpoint , to get [x,y,z,l,w,h]
    Desc: Given heatmap [B,H,W,C], find for each cell, which class is it, by taking the  maximum over classes. Then
    input
        predictions: [B , 5]    
        threshold: numerical thresh to determine if an object is centered at each x,y location in the BEV grid 
    """
    # for p in predictions.shape[0]: #iterate over batch
    heatmap = predictions["heatmap"]
    B, C , H , W = heatmap.shape 

    heatmap = heatmap.permute(0,2,3,1)
    heatmap = heatmap.reshape(B, H * W, C)

    scores , labels = torch.max(heatmap , dim = -1)   # Both have shape [B, 20000]

    candidates = []

    for b in range(B):
        keep = scores[b] > score_threshold

        # Flat indices of retained spatial cells, in those batches 
        indices = keep.nonzero(as_tuple=True)[0]

        cy = indices // W
        cx = indices % W

        candidates.append({
            "cx": cx,
            "cy": cy,
            "scores": scores[b][keep],
            "labels": labels[b][keep],
        })

    return candidates #[B , 4]

def get_decoded_bbox_from_pred(predictions, 
                               score_threshold , 
                               metres_per_cell ,
                               x_min, 
                                y_min,
                               filter=None)->List:
    """
    decode predictions to style [x,y,z,l,w,h,yaw], for all predictions inside a batch.
    To get x,y we change from output bev coords to real world meters. 
    Inputs
        predictions: pred of centerpoint in batches
        score_threshold: scoring threshold to get keypoints (obj centers)
    outputs
        candidates: batches of decoded predictions
    """

    candidates=decode_predictions(predictions=predictions,score_threshold=score_threshold)

    results = []
    for i , (batch_candidate) in enumerate(candidates):
        offsets=predictions["reg"] #select batch
        heights =predictions["height"]
        dimensions= predictions["dim"]
        rotations = predictions["rot"]
 
        #get candidate bbox centeres
        cy,cx=batch_candidate["cy"],batch_candidate["cx"] #here cy,cx are H*W dimensional

        #get candidate offsets and other attributes only at cx cy which have passed scoring threshold
        dx = offsets[i,0,cy,cx].float()
        dy = offsets[i,1,cy,cx].float()

        x=x_min + (cx + dx)*metres_per_cell 
        y=y_min + (cy + dy)*metres_per_cell
        z=heights[i,0,cy,cx]
        w=dimensions[i,0,cy,cx].float().exp()
        l=dimensions[i,1,cy,cx].float().exp()
        h=dimensions[i,2,cy,cx].float().exp()

        sin_yaw = rotations[i, 0, cy, cx].float()
        cos_yaw = rotations[i, 1, cy, cx].float()
        yaw = torch.atan2(sin_yaw, cos_yaw)

        boxes = torch.stack(
            [x, y, z,  l, w , h, yaw],
            dim=-1,
        )  # [N, 7], including [0, 7] when no candidates survive

        # if x.numel() > 0:
        #     print("Heatmap shape:", predictions["heatmap"].shape)
        #     print("Grid:", x_min, y_min, metres_per_cell)
        #     print("cx range:", cx.min().item(), cx.max().item())
        #     print("cy range:", cy.min().item(), cy.max().item())
        #     print("dx range:", dx.min().item(), dx.max().item())
        #     print("dy range:", dy.min().item(), dy.max().item())
        #     print("x metres:", x.min().item(), x.max().item())
        #     print("y metres:", y.min().item(), y.max().item())
        #     sys.exit(0)


        #results are actual predicted boxes decoded in metres, with scores (heatmap) 
        # at each H*W cell of the BEV grid
        results.append({
            "boxes": boxes,
            "scores": batch_candidate["scores"],
            "labels": batch_candidate["labels"],

        })

        # candidates[i].update({"cx":candidates[i].update({"cx"}) + offsets[i][i,0,cy,cx].float()})
        # candidates[i].update({"cy" : candidates[i].update({"cx"}) + offsets[i][i,1,cy,cx].float()})
        # candidates[i].update({"heights" : })
        # candidates[i].update({"dimensions" : })
        # candidates[i].update({"rots" : rotations[i,:,cy,cx]})

    
    return results



          

     