import torch
import torch.nn as nn
import torch.nn.init


class _FPNUp(nn.Module):

    def __init__(self, num_input_features, skip_channel_adjust=True):
        super().__init__()
        self.conv_channel_adjust = nn.Conv2d(num_input_features, 256,
            kernel_size=1)
        self.conv_fusion = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x, skip):
        upsample = nn.UpsamplingBilinear2d(size=skip.size()[2:])
        x = upsample(x)
        skip = self.conv_channel_adjust(skip)
        fused = self.conv_fusion(x + skip)
        return fused


def get_inputs():
    return [torch.rand([4, 256, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_input_features': 4}]
