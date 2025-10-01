import torch
import torch.nn as nn
import torch._utils


class AEBatch(nn.Module):

    def __init__(self):
        super(AEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_num):
        return torch.abs(torch.sum(estimated_density_map, dim=(1, 2, 3)) -
            gt_num)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
