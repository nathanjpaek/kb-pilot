import torch
import torch.nn as nn


class SC(nn.Module):

    def __init__(self):
        super(SC, self).__init__()
        kernel_size = 3
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(
            kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean
            (x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
