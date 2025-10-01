import torch
import torch.nn as nn
import torch.nn.functional as F


class discriminator(nn.Module):

    def __init__(self, d_dim, z_dim):
        super(discriminator, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.conv_1 = nn.Conv3d(1, self.d_dim, 4, stride=1, padding=0, bias
            =True)
        self.conv_2 = nn.Conv3d(self.d_dim, self.d_dim * 2, 3, stride=2,
            padding=0, bias=True)
        self.conv_3 = nn.Conv3d(self.d_dim * 2, self.d_dim * 4, 3, stride=1,
            padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.d_dim * 4, self.d_dim * 8, 3, stride=1,
            padding=0, bias=True)
        self.conv_5 = nn.Conv3d(self.d_dim * 8, self.d_dim * 16, 3, stride=
            1, padding=0, bias=True)
        self.conv_6 = nn.Conv3d(self.d_dim * 16, self.z_dim, 1, stride=1,
            padding=0, bias=True)

    def forward(self, voxels, is_training=False):
        out = voxels
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)
        out = self.conv_6(out)
        out = torch.sigmoid(out)
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {'d_dim': 4, 'z_dim': 4}]
