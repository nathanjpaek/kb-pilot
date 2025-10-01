import torch
from torch import nn


class ResidualAttention(nn.Module):

    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2d(in_channels=channel, out_channels=num_class,
            kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        _b, _c, _h, _w = x.shape
        y_raw = self.fc(x).flatten(2)
        y_avg = torch.mean(y_raw, dim=2)
        y_max = torch.max(y_raw, dim=2)[0]
        score = y_avg + self.la * y_max
        return score


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {}]
