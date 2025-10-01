import torch
from torch import nn
from torch.nn import Parameter


def positive(weight, cache=None):
    weight.data *= weight.data.ge(0).float()
    return cache


class WidthXHeightXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between three vectors over width,
    height and spatial.
    """

    def __init__(self, in_shape, outdims, components=1, bias=True,
        normalize=True, positive=False, width=None, height=None, eps=1e-06):
        super().__init__()
        self.in_shape = in_shape
        self.eps = eps
        c, w, h = self.in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.positive = positive
        self.components = components
        self.width = Parameter(torch.Tensor(self.outdims, 1, w, 1, components)
            ) if width is None else width
        self.height = Parameter(torch.Tensor(self.outdims, 1, 1, h, components)
            ) if height is None else height
        self.features = Parameter(torch.Tensor(self.outdims, c, 1, 1))
        assert self.width.size(4) == self.height.size(4
            ), 'The number of components in width and height do not agree'
        self.components = self.width.size(4)
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()

    def initialize(self, init_noise=0.001):
        self.width.data.normal_(0, init_noise)
        self.height.data.normal_(0, init_noise)
        self.features.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    @property
    def normalized_width(self):
        if self.positive:
            positive(self.width)
        if self.normalize:
            return self.width / (self.width.pow(2).sum(2, keepdim=True) +
                self.eps).sqrt().expand_as(self.width)
        else:
            return self.width

    @property
    def normalized_height(self):
        _c, _w, _h = self.in_shape
        if self.positive:
            positive(self.height)
        if self.normalize:
            return self.height / (self.height.pow(2).sum(3, keepdim=True) +
                self.eps).sqrt().expand_as(self.height)
        else:
            return self.height

    @property
    def spatial(self):
        _c, w, h = self.in_shape
        n, comp = self.outdims, self.components
        weight = self.normalized_width.expand(n, 1, w, h, comp
            ) * self.normalized_height.expand(n, 1, w, h, comp)
        weight = weight.sum(4, keepdim=True).view(n, 1, w, h)
        return weight

    @property
    def weight(self):
        c, w, h = self.in_shape
        n, _comp = self.outdims, self.components
        weight = self.spatial.expand(n, c, w, h) * self.features.expand(n,
            c, w, h)
        weight = weight.view(self.outdims, -1)
        return weight

    @property
    def basis(self):
        c, w, h = self.in_shape
        return self.weight.view(-1, c, w, h).data.cpu().numpy()

    def forward(self, x):
        N = x.size(0)
        y = x.view(N, -1) @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias.expand_as(y)
        return y

    def __repr__(self):
        return ('spatial positive ' if self.positive else '') + ('normalized '
             if self.normalize else ''
            ) + self.__class__.__name__ + ' (' + '{} x {} x {}'.format(*
            self.in_shape) + ' -> ' + str(self.outdims
            ) + ') spatial rank {}'.format(self.components)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_shape': [4, 4, 4], 'outdims': 4}]
