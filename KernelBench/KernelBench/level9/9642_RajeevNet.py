import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class RajeevNet(nn.Module):

    def __init__(self):
        super(RajeevNet, self).__init__()

    def forward(self, input):
        x = nn.AdaptiveAvgPool2d(1)(input)
        x = 20 * F.normalize(x)
        x = x.contiguous()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
