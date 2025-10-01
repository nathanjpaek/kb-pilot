import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):

    def __init__(self, channel=512, out_class=1, is_dis=True):
        super(Encoder, self).__init__()
        self.is_dis = is_dis
        self.channel = channel
        n_class = out_class
        self.conv1 = nn.Conv3d(1, channel // 8, kernel_size=4, stride=2,
            padding=1)
        self.conv2 = nn.Conv3d(channel // 8, channel // 4, kernel_size=4,
            stride=2, padding=1)
        self.bn2 = nn.InstanceNorm3d(channel // 4)
        self.conv3 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4,
            stride=2, padding=1)
        self.bn3 = nn.InstanceNorm3d(channel // 2)
        self.conv4 = nn.Conv3d(channel // 2, channel, kernel_size=4, stride
            =2, padding=1)
        self.bn4 = nn.InstanceNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1,
            padding=0)

    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        output = h5
        return output


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
