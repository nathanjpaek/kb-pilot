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


class Transformer(torch.nn.Module):
    """
    the min input size is [1, 3, 16, 16] as the kernel size and stride reduce the height and width
    otherwise exception might caused as the input_size != output_size
    """

    def __init__(self):
        super().__init__()
        self.conv1 = Convlayer(in_channels=3, out_channels=32, kernel_size=
            9, stride=1)
        self.inst1 = torch.nn.InstanceNorm2d(num_features=32, affine=True)
        self.conv2 = Convlayer(in_channels=32, out_channels=64, kernel_size
            =3, stride=2)
        self.inst2 = torch.nn.InstanceNorm2d(num_features=64, affine=True)
        self.conv3 = Convlayer(in_channels=64, out_channels=128,
            kernel_size=3, stride=2)
        self.inst3 = torch.nn.InstanceNorm2d(num_features=128, affine=True)
        self.res1 = Residential(128)
        self.res2 = Residential(128)
        self.res3 = Residential(128)
        self.res4 = Residential(128)
        self.res5 = Residential(128)
        self.upsample1 = UpsampleConvLayer(in_channels=128, out_channels=64,
            kernel_size=3, stride=1, upsample=2)
        self.upsample2 = UpsampleConvLayer(in_channels=64, out_channels=32,
            kernel_size=3, stride=1, upsample=2)
        self.upsample3 = UpsampleConvLayer(in_channels=32, out_channels=3,
            kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.inst1(self.conv1(x)))
        x = self.relu(self.inst2(self.conv2(x)))
        x = self.relu(self.inst3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.upsample1(x))
        x = self.relu(self.upsample2(x))
        x = self.relu(self.upsample3(x))
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
