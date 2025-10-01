import torch
import torch.nn as nn
import torch.nn.functional as F


class decoderVH(nn.Module):

    def __init__(self):
        super(decoderVH, self).__init__()
        self.dconv0 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn0 = nn.GroupNorm(8, 128)
        self.dconv1 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn1 = nn.GroupNorm(4, 64)
        self.dconv2 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size
            =3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = F.relu(self.dgn0(self.dconv0(x)), True)
        x = F.relu(self.dgn1(self.dconv1(F.interpolate(x, scale_factor=2,
            mode='bilinear'))), True)
        x = torch.tanh(self.dconv2(F.interpolate(x, scale_factor=2, mode=
            'bilinear')))
        mask, normal = torch.split(x, [1, 3], dim=1)
        mask = (mask + 1) * 0.5
        normal = normal / torch.clamp(torch.sqrt(torch.sum(normal * normal,
            dim=1)).unsqueeze(1), min=1e-06)
        return normal, mask


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
