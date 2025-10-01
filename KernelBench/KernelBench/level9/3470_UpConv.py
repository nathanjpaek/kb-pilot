import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip_connection):
        out = self.tconv(x)
        if out.shape != skip_connection.shape:
            out = TF.resize(out, size=skip_connection.shape[2:])
        out = torch.cat([skip_connection, out], axis=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
