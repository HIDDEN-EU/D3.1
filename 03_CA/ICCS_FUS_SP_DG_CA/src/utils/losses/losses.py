import torch 
import logging
from torchmetrics import Metric

logging.getLogger(__file__)

# class 3d_loss_metric()


class detection_3d_loss(torch.nn.Module):
    def __init__(self , device):
        super().__init__()
        self.device=device
        
    # def get_current_losses(self):
    #     return

    def forward(self, 
                pred , 
                target, 
                print_losses=True):
        mask = target["target_mask"]

        heatmap_loss = self.heatmap_loss(pred["heatmap"] , target["target_heatmap"] )
        offset_loss = self.regression_loss(pred["reg"] , target["target_offsets"] , mask )
        height_loss = self.regression_loss(pred["height"] , target["target_height"] , mask )
        size_loss = self.regression_loss(pred["dim"] , target["target_size"] , mask )
        rots_loss = self.regression_loss(pred["rot"] , target["target_rots"] , mask)

        tot_loss = heatmap_loss+offset_loss+height_loss+size_loss+rots_loss

        if print_losses:
            print("Heatmap loss is {}".format(heatmap_loss))
            print("Offset loss is {}".format(offset_loss))
            print("height loss is {}".format(height_loss))
            print("size loss is {}".format(size_loss))
            print("Rotation loss is {}".format(rots_loss))


        return tot_loss,heatmap_loss,offset_loss,height_loss,size_loss,rots_loss

    
    def regression_loss(self,pred, target, mask):
        """
        loss for size,offset,rotation,
        """
        valid = mask[:, 0].to(self.device,
                              dtype=torch.bool)  # [B, H, W]

        pred = pred.permute(0, 2, 3, 1)      # [B, H, W, channels]
        target = target.permute(0, 2, 3, 1)

        if not valid.any():
            return pred.sum() * 0.0

        return torch.nn.functional.l1_loss(
            pred[valid],
            target[valid],
        )

    def heatmap_loss(self, pred, target):
        # pred already has sigmoid applied.
        p = pred.float().clamp(1e-4, 1 - 1e-4)
        target = target.float()

        # Temporary debugging checks
        if not torch.isfinite(p).all():
            raise RuntimeError("Heatmap predictions contain NaN/Inf")
        if not torch.isfinite(target).all():
            raise RuntimeError("Heatmap targets contain NaN/Inf")
        if ((target < 0) | (target > 1)).any():
            raise RuntimeError("Heatmap targets must be in [0, 1]")

        logging.info("Prediction dtype:{}".format(pred.dtype))
        logging.info("Positive centers:{}".format(target.eq(1).sum().item()))
        logging.info("Mean probability:{}".format(pred.detach().float().mean().item()))

        positive = target.eq(1).to(p.dtype)
        negative = target.lt(1).to(p.dtype)

        positive_loss = (
            -torch.log(p)
            * (1 - p).pow(2)
            * positive
        )

        negative_loss = (
            -torch.log(1 - p)
            * p.pow(2)
            * (1 - target).pow(4)
            * negative
        )

        num_centres = positive.sum().clamp_min(1)
        return (positive_loss.sum() + negative_loss.sum()) / num_centres