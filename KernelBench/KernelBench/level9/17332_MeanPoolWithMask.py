import torch
from torch import nn
import torch.utils.data


class MeanPoolWithMask(nn.Module):

    def __init__(self):
        super(MeanPoolWithMask, self).__init__()
        self.inf = 10000000000000.0

    def forward(self, tensor, mask, dim=0):
        masks = mask.view(mask.size(0), mask.size(1), -1).float()
        return torch.sum(tensor * masks, dim=dim) / torch.sum(masks, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 16]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
