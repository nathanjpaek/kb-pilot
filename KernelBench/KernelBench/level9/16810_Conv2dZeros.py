import torch
import torch.nn as nn
import torch.utils.data


def cpd_mean(tensor, dim=None, keepdims=False):
    if dim is None:
        return tensor.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


class Actnormlayer(nn.Module):

    def __init__(self, num_features, scale=1.0):
        super(Actnormlayer, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-06

    def initialize_parameters(self, x):
        if not self.training:
            return
        with torch.no_grad():
            bias = -cpd_mean(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = cpd_mean((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.0

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()
        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj
        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)
        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        return x, ldj


class Conv2d(nn.Conv2d):
    pad_dict = {'same': lambda kernel, stride: [(((k - 1) * s + 1) // 2) for
        k, s in zip(kernel, stride)], 'valid': lambda kernel, stride: [(0) for
        _ in kernel]}

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError('{} is not supported'.format(padding))
        return padding

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3],
        stride=[1, 1], padding='same', do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
            stride, padding, bias=not do_actnorm)
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = Actnormlayer(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super(Conv2d, self).forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x, 0.0)
        return x


class Conv2dZeros(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3],
        stride=[1, 1], padding='same', logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super(Conv2dZeros, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding)
        self.logscale_factor = logscale_factor
        self.register_parameter('logs', nn.Parameter(torch.zeros(
            out_channels, 1, 1)))
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super(Conv2dZeros, self).forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
