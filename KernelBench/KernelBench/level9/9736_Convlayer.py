import torch


class Convlayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.refl = torch.nn.ReflectionPad2d(padding)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            stride)

    def forward(self, x):
        x = self.refl(x)
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
