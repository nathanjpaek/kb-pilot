import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed


class CombinedPooling(nn.Module):

    def __init__(self):
        super().__init__()
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_pooled = self.max_pooling(x)
        avg_pooled = self.avg_pooling(x)
        return max_pooled + avg_pooled


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
