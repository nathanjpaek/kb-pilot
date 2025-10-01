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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channel': 4}]
