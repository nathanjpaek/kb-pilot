import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, 'Please assign weight and bias before calling AdaIN!'
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.
            weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class ResBlock(nn.Module):

    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        padding = 1
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, bias=True)
        if norm == 'in':
            self.norm1 = nn.InstanceNorm2d(dim)
        elif norm == 'adain':
            self.norm1 = AdaptiveInstanceNorm2d(dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, bias=True)
        if norm == 'in':
            self.norm2 = nn.InstanceNorm2d(dim)
        elif norm == 'adain':
            self.norm2 = AdaptiveInstanceNorm2d(dim)

    def forward(self, x):
        residual = x
        x = self.conv1(self.pad(x))
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(self.pad(x))
        out = self.norm2(x)
        out += residual
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
