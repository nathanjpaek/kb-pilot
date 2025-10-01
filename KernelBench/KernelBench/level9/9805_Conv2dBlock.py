import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveInstanceLayerNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.9,
        using_moving_average=True, using_bn=False):
        super(AdaptiveInstanceLayerNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features
        if self.using_bn:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(3)
            self.rho[:, :, 1].data.fill_(1)
            self.rho[:, :, 2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1,
                num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features,
                1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(3.2)
            self.rho[:, :, 1].data.fill_(1)
        self.weight = None
        self.bias = None

    def forward(self, input):
        assert self.weight is not None and self.bias is not None, 'Please assign AdaILN weight first'
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True
            ), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True
            ), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)
        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True
                    ), torch.var(input, dim=[0, 2, 3], keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln
        out = out * self.weight.unsqueeze(2).unsqueeze(3
            ) + self.bias.unsqueeze(2).unsqueeze(3)
        return out


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
        assert self.weight is not None and self.bias is not None, 'Please assign AdaIN weight first'
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.
            weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Identity(nn.Module):

    def forward(self, x):
        return x


class InstanceLayerNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.9,
        using_moving_average=True, using_bn=False):
        super(InstanceLayerNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features
        if self.using_bn:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3)
            self.rho[:, :, 2].data.fill_(3)
            self.register_buffer('running_mean', torch.zeros(1,
                num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features,
                1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3.2)
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True
            ), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True
            ), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)
        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True
                    ), torch.var(input, dim=[0, 2, 3], keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1
            ) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class Conv2dBlock(nn.Module):

    def __init__(self, in_dim, out_dim, ks, st, padding=0, norm='none',
        activation='relu', pad_type='zero', use_bias=True, activation_first
        =False, groups=1, sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        if padding == 0:
            self.pad = Identity()
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(4, norm_dim, 0.8)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'iln':
            self.norm = InstanceLayerNorm2d(norm_dim)
        elif norm == 'adailn':
            self.norm = AdaptiveInstanceLayerNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if sn:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim,
                ks, st, bias=self.use_bias, groups=groups))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.
                use_bias, groups=groups)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'ks': 4, 'st': 4}]
