import torch
import torch.nn as nn
import torch.multiprocessing
import torch.onnx


class MS_Block(nn.Module):

    def __init__(self, input_feature, out_feature, d=[1, 2, 4], group=1):
        super(MS_Block, self).__init__()
        self.l1 = nn.Conv2d(input_feature, out_feature, 3, padding=d[0],
            dilation=d[0], bias=False, groups=group)
        self.l2 = nn.Conv2d(input_feature, out_feature, 3, padding=d[1],
            dilation=d[1], bias=False, groups=group)
        self.l3 = nn.Conv2d(input_feature, out_feature, 3, padding=d[2],
            dilation=d[2], bias=False, groups=group)

    def forward(self, x):
        out = self.l1(x) + self.l2(x) + self.l3(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_feature': 4, 'out_feature': 4}]
