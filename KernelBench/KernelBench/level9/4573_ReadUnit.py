import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin


class ReadUnit(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memories, k, c):
        """
        :param memories:
        :param k: knowledge
        :param c: control
        :return: r_i
        """
        m_prev = memories[-1]
        I = self.mem(m_prev).unsqueeze(2) * k
        I = self.concat(torch.cat([I, k], 1).permute(0, 2, 1))
        attn = I * c[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)
        read = (attn * k).sum(2)
        return read


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
