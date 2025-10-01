import torch
import torch.nn as nn
from torch.distributions import transforms as transform


class Flow(transform.Transform, nn.Module):
    """
    Main class for a single flow.
    """

    def __init__(self, amortized='none'):
        """ Initialize as both transform and module """
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
        self.amortized = amortized

    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.001, 0.001)

    def __hash__(self):
        """ Dirty hack to ensure nn.Module compatibility """
        return nn.Module.__hash__(self)

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        pass

    def n_parameters(self):
        """ Return number of parameters in flow """
        return 0


class ActNormFlow(Flow):
    """
    An implementation of the activation normalization layer defined in
    Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, dim, amortized='none'):
        super(ActNormFlow, self).__init__()
        self.weight = []
        self.bias = []
        self.amortized = amortized
        self.weight = amortized_ones(amortized, (1, dim, 1, 1))
        self.bias = amortized_zeros(amortized, (1, dim, 1, 1))
        self.initialized = False
        self.dim = dim

    def _call(self, z):
        return z * torch.exp(self.weight) + self.bias

    def _inverse(self, z):
        return (z - self.bias) * torch.exp(-self.weight)

    def log_abs_det_jacobian(self, z):
        if self.initialized is False:
            self.bias.data.copy_(z.mean((0, 2, 3), keepdim=True) * -1)
            self.weight.data.copy_(torch.log(1.0 / (torch.sqrt(((z + self.
                bias.data) ** 2).mean((0, 2, 3), keepdim=True)) + 1e-06)))
            self.initialized = True
        return torch.sum(self.weight).repeat(z.shape[0], 1) * z.shape[2
            ] * z.shape[3]

    def set_parameters(self, params, batch_dim):
        """ Set parameters values (sub-modules) """
        if self.amortized != 'none':
            self.weight = params[:, :self.dim ** 2]
            self.bias = params[:, self.dim ** 2:self.dim ** 2 * 2]

    def n_parameters(self):
        """ Return number of parameters in flow """
        return self.dim * 2


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
        stride=[1, 1], padding='same', do_actnorm=False, weight_std=0.001):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
            padding, bias=not do_actnorm)
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNormFlow(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x = self.actnorm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
