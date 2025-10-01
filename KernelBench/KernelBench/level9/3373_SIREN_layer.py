import torch
import numpy as np
import torch.nn as nn


def act(act_fun='LeakyReLU'):
    """
        Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun[:3] == 'ELU':
            if len(act_fun) > 3:
                param = float(act_fun[3:])
                return nn.ELU(param, inplace=True)
            return nn.ELU(inplace=True)
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'tanh':
            return Tanh()
        elif act_fun == 'sine':
            return Sin()
        elif act_fun == 'soft':
            return nn.Softplus()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


class Tanh(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x)


class Sin(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class SIREN_layer(nn.Module):

    def __init__(self, ch_in, ch_out, frist=False, act_fun='sine', omega_0=30):
        super(SIREN_layer, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias
            =True)
        self.act_fun = act(act_fun)
        self.omega_0 = omega_0
        self.in_features = ch_in
        self.frist = frist
        self.init()

    def init(self):
        with torch.no_grad():
            if self.frist:
                self.conv1.weight.uniform_(-1 / self.in_features, 1 / self.
                    in_features)
            else:
                self.conv1.weight.uniform_(-np.sqrt(6 / self.in_features) /
                    self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        x = self.conv1(x)
        return self.act_fun(self.omega_0 * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ch_in': 4, 'ch_out': 4}]
