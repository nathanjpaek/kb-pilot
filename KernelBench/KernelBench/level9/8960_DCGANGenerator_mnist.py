import functools
import torch
import torch.utils.data
import torch
import torch.nn as nn


class DCGANGenerator_mnist(nn.Module):

    def __init__(self, z_dim, ngf=64, output_nc=1, norm_layer=nn.BatchNorm2d):
        super(DCGANGenerator_mnist, self).__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        self.img_size = 28 * 28 * output_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.fc1 = nn.Linear(self.z_dim, 256, bias=use_bias)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features *
            2, bias=use_bias)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features *
            2, bias=use_bias)
        self.fc4 = nn.Linear(self.fc3.out_features, self.img_size, bias=
            use_bias)
        self.model = nn.Sequential(self.fc1, nn.LeakyReLU(negative_slope=
            0.2), self.fc2, nn.LeakyReLU(negative_slope=0.2), self.fc3, nn.
            LeakyReLU(negative_slope=0.2), self.fc4, nn.Tanh())

    def forward(self, noise):
        img = self.model(noise.view(noise.shape[0], -1))
        return img.view(noise.shape[0], 1, 28, 28)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4}]
