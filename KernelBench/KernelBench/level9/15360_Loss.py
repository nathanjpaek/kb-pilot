import torch
import torch.nn as nn
import torch.utils.data


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, gt_region, gt_affinity, pred_region, pred_affinity,
        conf_map):
        loss = torch.mean(((gt_region - pred_region).pow(2) + (gt_affinity -
            pred_affinity).pow(2)) * conf_map)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
