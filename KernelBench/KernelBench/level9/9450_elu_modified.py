import torch
import torch.nn as nn
import torch.utils.data


class elu_modified(nn.Module):

    def __init__(self, alpha=1.0, shift=5.0, epsilon=1e-07):
        super(elu_modified, self).__init__()
        self.alpha = alpha
        self.shift = shift
        self.epsilon = epsilon
        self.elu = nn.ELU(alpha=alpha)

    def forward(self, x):
        return self.elu(x + self.shift) + 1.0 + self.epsilon


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
