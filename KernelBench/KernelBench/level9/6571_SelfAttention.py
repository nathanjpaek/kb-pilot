import math
import torch
import torch.nn as nn
from torch.nn import init


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0
            ) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun


class SelfAttention(nn.Module):

    def __init__(self, feat_dim, proj_dim, gamma):
        super(SelfAttention, self).__init__()
        self.W_g = torch.nn.Conv2d(feat_dim, proj_dim, kernel_size=(1, 1))
        self.W_f = torch.nn.Conv2d(feat_dim, proj_dim, kernel_size=(1, 1))
        self.W_h = torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=(1, 1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.gamma = gamma
        self.W_g.apply(weights_init('gaussian'))
        self.W_f.apply(weights_init('gaussian'))
        self.W_h.apply(weights_init('gaussian'))
        self.W_g
        self.W_f
        self.W_h

    def forward(self, x):
        f = self.W_f.forward(x)
        g = self.W_g.forward(x)
        h = self.W_h.forward(x)
        N, feat_D, hgt, wid = x.size(0), x.size(1), x.size(2), x.size(3)
        proj_D = f.size(1)
        f = f.view(N, proj_D, -1).transpose(1, 2)
        g = g.view(N, proj_D, -1).transpose(1, 2)
        h = h.view(N, feat_D, -1).transpose(1, 2)
        o = []
        for idx in range(N):
            aff = torch.mm(g[idx], f[idx].transpose(0, 1))
            aff = self.softmax(aff)
            cur_o = torch.mm(aff, h[idx])
            cur_o = cur_o.transpose(0, 1).contiguous()
            cur_o = cur_o.view(1, feat_D, hgt, wid)
            o.append(cur_o)
        o = torch.cat(o, 0)
        y = self.gamma * o + (1 - self.gamma) * x
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feat_dim': 4, 'proj_dim': 4, 'gamma': 4}]
