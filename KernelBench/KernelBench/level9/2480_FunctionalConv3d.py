import torch


class FunctionalConv3d(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv3d(*args, **kwargs)

    def forward(self, x):
        x = torch.nn.functional.conv3d(x, self.conv.weight, self.conv.bias,
            self.conv.stride, self.conv.padding, self.conv.dilation, self.
            conv.groups)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
