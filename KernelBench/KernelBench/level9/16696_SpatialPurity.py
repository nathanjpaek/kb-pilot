import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils
import torch.distributed


class SpatialPurity(nn.Module):

    def __init__(self, in_channels=19, padding_mode='zeros', size=3):
        super(SpatialPurity, self).__init__()
        assert size % 2 == 1, 'error size'
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            in_channels, kernel_size=size, stride=1, padding=int(size / 2),
            bias=False, padding_mode=padding_mode, groups=in_channels)
        a = torch.ones((size, size), dtype=torch.float32)
        a = a.unsqueeze(dim=0).unsqueeze(dim=0)
        a = a.repeat([in_channels, 1, 1, 1])
        a = nn.Parameter(a)
        self.conv.weight = a
        self.conv.requires_grad_(False)

    def forward(self, x):
        summary = self.conv(x)
        count = torch.sum(summary, dim=1, keepdim=True)
        dist = summary / count
        spatial_purity = torch.sum(-dist * torch.log(dist + 1e-06), dim=1,
            keepdim=True)
        return spatial_purity


def get_inputs():
    return [torch.rand([4, 19, 64, 64])]


def get_init_inputs():
    return [[], {}]
