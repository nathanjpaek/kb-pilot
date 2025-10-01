import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):

    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in):
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score).view(x_in.size(0),
            x_in.size(1), 1)
        scored_x = x_in * attention_score
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x, attention_score


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'attention_size': 4}]
