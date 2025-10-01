import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSoftMax(nn.Module):

    def __init__(self):
        super(LogSoftMax, self).__init__()

    def forward(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
