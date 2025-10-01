import torch
import torch.nn as nn
import torch.utils.data.distributed


class TLU(nn.Module):
    """ Thresholded Linear Unit """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return torch.max(x, self.tau)

    def extra_repr(self):
        return 'num_features={}'.format(self.num_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
