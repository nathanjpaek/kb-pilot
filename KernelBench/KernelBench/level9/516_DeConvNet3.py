import torch
import torch.nn as nn


def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'spherical':
        return SphericalActivation()
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


class SphericalActivation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class DeConvNet3(nn.Module):

    def __init__(self, in_chan=1, out_chan=1, nh=32, out_activation=
        'linear', activation='relu', num_groups=None):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet3, self).__init__()
        self.num_groups = num_groups
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias
            =True)
        self.conv1 = nn.ConvTranspose2d(nh * 32, nh * 16, kernel_size=4,
            stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=4,
            stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, out_chan, kernel_size=1,
            bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        layers = [self.fc1]
        layers += [] if self.num_groups is None else [self.get_norm_layer(
            nh * 32)]
        layers += [get_activation(activation), self.conv1]
        layers += [] if self.num_groups is None else [self.get_norm_layer(
            nh * 16)]
        layers += [get_activation(activation), self.conv2]
        layers += [] if self.num_groups is None else [self.get_norm_layer(
            nh * 8)]
        layers += [get_activation(activation), self.conv3]
        out_activation = get_activation(out_activation)
        if out_activation is not None:
            layers.append(out_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_norm_layer(self, num_channels):
        if self.num_groups is not None:
            return nn.GroupNorm(num_groups=self.num_groups, num_channels=
                num_channels)
        else:
            return None


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {}]
