import torch
import torch.nn as nn


class CAMBlock(nn.Module):

    def __init__(self):
        super(CAMBlock, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_tmp = x.permute([0, 2, 1])
        maxpool_output = self.maxpool(x_tmp)
        avgpool_output = self.avgpool(x_tmp)
        x_tmp = torch.cat([maxpool_output, avgpool_output], dim=-1)
        x_tmp = x_tmp.permute([0, 2, 1])
        x_tmp = self.sigmoid(self.conv(x_tmp))
        return x * x_tmp


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
