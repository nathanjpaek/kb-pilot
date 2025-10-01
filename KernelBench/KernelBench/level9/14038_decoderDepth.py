import torch
import torch.nn as nn
import torch.nn.functional as F


class decoderDepth(nn.Module):

    def __init__(self):
        super(decoderDepth, self).__init__()
        self.dconv0 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn0 = nn.GroupNorm(32, 512)
        self.dconv1 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn1 = nn.GroupNorm(16, 256)
        self.dconv2 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn2 = nn.GroupNorm(16, 256)
        self.dconv3 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(8, 128)
        self.dconv4 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn4 = nn.GroupNorm(8, 128)
        self.dconv5 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn5 = nn.GroupNorm(4, 64)
        self.convFinal = nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x1 = F.relu(self.dgn0(self.dconv0(x)), True)
        x2 = F.relu(self.dgn1(self.dconv1(x1)), True)
        x3 = F.relu(self.dgn2(self.dconv2(F.interpolate(x2, scale_factor=2,
            mode='bilinear'))), True)
        x4 = F.relu(self.dgn3(self.dconv3(x3)), True)
        x5 = F.relu(self.dgn4(self.dconv4(F.interpolate(x4, scale_factor=2,
            mode='bilinear'))), True)
        x6 = F.relu(self.dgn5(self.dconv5(x5)), True)
        depthDelta = 0.5 * torch.tanh(self.convFinal(F.interpolate(x6,
            scale_factor=2, mode='bilinear')))
        return depthDelta


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {}]
