import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.checkpoint


class LinearAttention(nn.Module):

    def __init__(self, in_size):
        super(LinearAttention, self).__init__()
        self.out = nn.Linear(in_size, 1)
        nn.init.orthogonal_(self.out.weight.data)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        self.alpha = self.softmax(self.out(input))
        x = (self.alpha.expand_as(input) * input).sum(dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
