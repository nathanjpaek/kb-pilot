import torch
import torch.nn as nn
import torch._utils


class SEBatch(nn.Module):

    def __init__(self):
        super(SEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_num):
        return torch.pow(torch.sum(estimated_density_map, dim=(1, 2, 3)) -
            gt_num, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
