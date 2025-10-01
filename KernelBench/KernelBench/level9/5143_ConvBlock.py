import torch


class ResBlock(torch.nn.Module):

    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=
            3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=
            3, stride=1, padding=1)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        out = x
        out = self.leaky_relu(out)
        out = self.conv1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        return out + x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channel, num_channel):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channel, num_channel, kernel_size=3,
            stride=1)
        self.max = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.res1 = ResBlock(num_channel=num_channel)
        self.res2 = ResBlock(num_channel=num_channel)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.max(out)
        out = self.res1(out)
        out = self.res2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'num_channel': 4}]
