import torch
import torch.utils.data
import torch.nn as nn


class L2NormLoss(nn.Module):

    def __init__(self):
        super(L2NormLoss, self).__init__()

    def forward(self, x1, x2, y1, y2):
        dist_in = torch.norm(x1 - x2, dim=1, keepdim=True)
        dist_out = torch.norm(y1 - y2, dim=1, keepdim=True)
        loss = torch.norm(dist_in - dist_out) / x1.size(0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
