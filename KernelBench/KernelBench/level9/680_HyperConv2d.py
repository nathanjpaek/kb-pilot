import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation
    mapper = {nn.LeakyReLU: 'leaky_relu', nn.ReLU: 'relu', nn.Tanh: 'tanh',
        nn.Sigmoid: 'sigmoid', nn.Softmax: 'sigmoid', nn.ELU: 'elu'}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k
    raise ValueError('Unkown given activation type : {}'.format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1
    activation_name = get_activation_name(activation)
    param = (None if activation_name != 'leaky_relu' else activation.
        negative_slope)
    gain = nn.init.calculate_gain(activation_name, param)
    return gain


def linear_init(layer, activation='relu'):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight
    if activation is None:
        return nn.init.xavier_uniform_(x)
    activation_name = get_activation_name(activation)
    if activation_name == 'leaky_relu':
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name in ['relu', 'elu']:
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ['sigmoid', 'tanh']:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)


class HyperConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0,
        dilation=1, groups=1, bias=True, transpose=False):
        super(HyperConv2d, self).__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, 'dim_in and dim_out must both be divisible by groups.'
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose
        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.
            ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(self.dim_in, self.dim_out //
                self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(self.dim_out, self.dim_in //
                self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(x, weight=weight, bias=bias, stride=self.stride,
            padding=self.padding, groups=self.groups, dilation=self.dilation)


def get_inputs():
    return [torch.rand([1, 1]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
