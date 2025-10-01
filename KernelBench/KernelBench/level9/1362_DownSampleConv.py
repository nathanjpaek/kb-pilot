import torch
import torch.nn as nn


class DownSampleConv(nn.Module):

    def __init__(self, in_feature, out_feature, kernel):
        super(DownSampleConv, self).__init__()
        self.conv = nn.Conv1d(in_feature, out_feature, kernel_size=kernel,
            stride=2, padding=kernel // 2, groups=in_feature)
        self._init_weigth()

    def _init_weigth(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        return nn.ReLU()(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4, 'kernel': 4}]
