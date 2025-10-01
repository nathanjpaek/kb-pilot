import torch
import torch.nn as nn


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=3, multi_branch=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.multi_branch = multi_branch
        if multi_branch:
            self.conv2 = nn.Conv2d(2, 1, 5, padding=2, bias=False)
            self.conv3 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        if not self.multi_branch:
            x = self.conv1(x)
        else:
            x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return self.sigmoid(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
