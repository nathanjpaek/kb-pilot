from torch.nn import Module
import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter


def clip_grad(v, min, max):
    v_tmp = v.expand_as(v)
    v_tmp.register_hook(lambda g: g.clamp(min, max))
    return v_tmp


class RNNCellBase(Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class GRUCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, grad_clip=None):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grad_clip = grad_clip
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh_rz = Parameter(torch.Tensor(2 * hidden_size,
            hidden_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        ih = F.linear(input, self.weight_ih, self.bias)
        hh_rz = F.linear(h, self.weight_hh_rz)
        if self.grad_clip:
            ih = clip_grad(ih, -self.grad_clip, self.grad_clip)
            hh_rz = clip_grad(hh_rz, -self.grad_clip, self.grad_clip)
        r = F.sigmoid(ih[:, :self.hidden_size] + hh_rz[:, :self.hidden_size])
        i = F.sigmoid(ih[:, self.hidden_size:self.hidden_size * 2] + hh_rz[
            :, self.hidden_size:])
        hhr = F.linear(h * r, self.weight_hh)
        if self.grad_clip:
            hhr = clip_grad(hhr, -self.grad_clip, self.grad_clip)
        n = F.relu(ih[:, self.hidden_size * 2:] + hhr)
        h = (1 - i) * n + i * h
        return h


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
