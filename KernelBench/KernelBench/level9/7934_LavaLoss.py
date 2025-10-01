import torch
import torch.nn.functional as F
import torch.nn as nn


class LavaLoss(nn.Module):
    """
    Depth gradient Loss for instance segmentation
    """

    def __init__(self):
        super(LavaLoss, self).__init__()
        pass

    def forward(self, seg_masks, gradient_map):
        gt_size = gradient_map.shape[1:]
        seg_masks = F.interpolate(seg_masks.unsqueeze(0), size=gt_size,
            mode='bilinear').squeeze(0)
        lava_loss_per_img = seg_masks.mul(gradient_map)
        loss = lava_loss_per_img.sum() / (gradient_map.sum() * seg_masks.
            shape[0])
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
