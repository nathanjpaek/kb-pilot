import torch
import torch.nn as nn


class CompressChannels(nn.Module):
    """
        Compresses the input channels to 2 by concatenating the results of
        Global Average Pooling(GAP) and Global Max Pooling(GMP).
        HxWxC => HxWx2

    """

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1)
            .unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):
    """

    Spatial Attention: 

                    HxWxC
                      |
                  ---------
                  |       |
                 GAP     GMP
                  |       |
                  ----C---
                      |
                    HxWx2
                      |
                    Conv
                     |
                  Sigmoid
                     |
                   HxWx1
                   
    Multiplying HxWx1 with input again gives output -> HxWxC

    """

    def __init__(self):
        super().__init__()
        self.compress_channels = CompressChannels()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5,
            stride=1, padding=2)

    def forward(self, x):
        compress_x = self.compress_channels(x)
        x_out = self.conv(compress_x)
        scale = torch.sigmoid(x_out)
        return x * scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
