import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.multiprocessing


class HeatmapLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = (pred - gt) ** 2 * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
