import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class StochasticPool2D(nn.Module):
    """
    Args: 
        kernel_size : size of the pooling kernel
        stride      : pool stride
    
    Note: valid padding is implemented
    """

    def __init__(self, kernel_size=3, stride=1, training=True):
        super(StochasticPool2D, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.training = training

    def forward(self, x, training=True):
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1],
            self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,))
        activ = F.relu(x)
        prob = torch.div(activ, activ.sum(dim=-1).unsqueeze_(-1))
        prob[prob != prob] = 1
        idx = torch.distributions.categorical.Categorical(prob).sample()
        out = x.gather(-1, idx.unsqueeze(-1))
        out = out.sum(dim=-1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
