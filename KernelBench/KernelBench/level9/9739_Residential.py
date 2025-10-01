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


class Residential(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = Convlayer(in_channels=in_channels, out_channels=
            in_channels, kernel_size=3)
        self.inst1 = torch.nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = Convlayer(in_channels=in_channels, out_channels=
            in_channels, kernel_size=3)
        self.inst2 = torch.nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        resident = x
        x = self.relu(self.inst1(self.conv1(x)))
        x = self.inst2(self.conv2(x))
        return resident + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
