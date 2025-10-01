import torch
from torch import nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    """
    Self attention layer: aggreagating a sequence into a single vector.
    This implementation uses the attention formula proposed by  Sukhbaatar etal. 2015
    https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf

    Usage:
    seq_len=10; bsz=16; in_dim=128
    attn = SelfAtnn(in_dim)
    x = torch.rand(seq_len, bsz, in_dim)  # 10x16x128
    y, a = attn(x)  # output y 16x128, attention weight a 10x16
    """

    def __init__(self, d_input, units=None):
        """
        :param d_input: input feature dimension
        :param units: dimension of internal projection, if None it will be set to d_input
        """
        super(SelfAttn, self).__init__()
        self.d_input = d_input
        self.units = units if units else d_input
        self.projection = nn.Linear(self.d_input, self.units)
        self.V = nn.Parameter(torch.Tensor(self.units, 1))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.projection.bias.data.zero_()
        self.projection.weight.data.uniform_(-initrange, initrange)
        self.V.data.uniform_(-initrange, initrange)

    def forward(self, x, mask=None):
        """
        ui = tanh(xW+b)
        a = softmax(uV)
        o = sum(a*x)
        :param x: input tensor [seq_len, bsz, feat_dim]
        :return:  output tensor [bsz, feat_dim]
        """
        ui = torch.tanh(self.projection(x))
        ai = F.softmax(torch.matmul(ui, self.V), dim=0)
        if mask is not None:
            ai = ai * mask.unsqueeze(-1)
            ai = ai / ai.sum(dim=0, keepdim=True)
        o = torch.sum(x * ai, dim=0)
        return o, ai.squeeze(-1)

    def extra_repr(self):
        return 'Sx?x%d -> ?x%d' % (self.d_input, self.d_input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_input': 4}]
