import torch


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        upsample=None):
        super().__init__()
        self.upsample = upsample
        reflectpad = kernel_size // 2
        self.reflectionpad = torch.nn.ReflectionPad2d(reflectpad)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            stride)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=self.
                upsample, mode='nearest')
        x = self.reflectionpad(x)
        return self.relu(self.conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1}]
