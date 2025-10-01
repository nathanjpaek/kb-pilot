import torch


class OutputBlock(torch.nn.Module):
    """Flatten output channels using 1x1x1 convolutions"""

    def __init__(self, ks, channels_in, channels_out):
        super(OutputBlock, self).__init__()
        self.convflat = torch.nn.Conv3d(in_channels=channels_in,
            out_channels=channels_out, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.convflat(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ks': 4, 'channels_in': 4, 'channels_out': 4}]
