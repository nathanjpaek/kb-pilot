import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenPoseLoss(nn.Module):

    def __init__(self):
        super(OpenPoseLoss, self).__init__()

    def forward(self, saved_for_loss, heatmap_target, heat_mask, paf_target,
        paf_mask):
        """
        tính loss
        Parameters
        ----------
        saved_for_loss : Output ofOpenPoseNet (list)

        heatmap_target : [num_batch, 19, 46, 46]
            Anotation information

        heatmap_mask : [num_batch, 19, 46, 46]
            

        paf_target : [num_batch, 38, 46, 46]
            PAF Anotation

        paf_mask : [num_batch, 38, 46, 46]
            PAF mask

        Returns
        -------
        loss : 
        """
        total_loss = 0
        for j in range(6):
            pred1 = saved_for_loss[2 * j] * paf_mask
            gt1 = paf_target.float() * paf_mask
            pred2 = saved_for_loss[2 * j + 1] * heat_mask
            gt2 = heatmap_target.float() * heat_mask
            total_loss += F.mse_loss(pred1, gt1, reduction='mean'
                ) + F.mse_loss(pred2, gt2, reduction='mean')
        return total_loss


def get_inputs():
    return [torch.rand([12, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand
        ([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
