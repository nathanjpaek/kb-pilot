import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.multiprocessing


def weights_init(init_type='gaussian'):

    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0
            ) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_func


class GaussianActivation(nn.Module):

    def __init__(self, a, mu, gamma_l, gamma_r):
        super(GaussianActivation, self).__init__()
        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.gamma_l = Parameter(torch.tensor(gamma_l, dtype=torch.float32))
        self.gamma_r = Parameter(torch.tensor(gamma_r, dtype=torch.float32))

    def forward(self, input_features):
        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.gamma_l.data = torch.clamp(self.gamma_l.data, 0.5, 2.0)
        self.gamma_r.data = torch.clamp(self.gamma_r.data, 0.5, 2.0)
        left = input_features < self.mu
        right = input_features >= self.mu
        g_A_left = self.a * torch.exp(-self.gamma_l * (input_features -
            self.mu) ** 2)
        g_A_left.masked_fill_(right, 0.0)
        g_A_right = 1 + (self.a - 1) * torch.exp(-self.gamma_r * (
            input_features - self.mu) ** 2)
        g_A_right.masked_fill_(left, 0.0)
        g_A = g_A_left + g_A_right
        return g_A


class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()
        self.func = nn.ReLU(True)
        self.alpha = alpha

    def forward(self, input_masks):
        return torch.pow(self.func(input_masks), self.alpha)


class ReverseAttentionLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
        padding=1, dilation=1, groups=1, bias=False):
        super(ReverseAttentionLayer, self).__init__()
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.mask_conv.apply(weights_init())
        self.gaussian_activation = GaussianActivation(a=1.1, mu=2.0,
            gamma_l=1.0, gamma_r=1.0)
        self.mask_update = MaskUpdate(alpha=0.8)

    def forward(self, input_masks):
        conv_masks = self.mask_conv(input_masks)
        gaussian_masks = self.gaussian_activation(conv_masks)
        output_masks = self.mask_update(conv_masks)
        return output_masks, gaussian_masks


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
