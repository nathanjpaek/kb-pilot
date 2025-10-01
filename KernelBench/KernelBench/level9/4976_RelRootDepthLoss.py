import torch
import torch.utils.data
import torch.nn as nn


class RelRootDepthLoss(nn.Module):

    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = torch.abs(root_depth_out - root_depth_gt) * root_valid
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
