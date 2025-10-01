import torch
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2,
            kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {}]
