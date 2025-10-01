import torch
import numpy as np
import torch.nn as nn


class GeometricLoss(nn.Module):

    def __init__(self, num_parameters=2, init=[0.0, -3.0]):
        self.num_parameters = num_parameters
        super(GeometricLoss, self).__init__()
        assert len(init) == num_parameters
        self.weight = nn.Parameter(torch.Tensor(np.array(init)))

    def forward(self, inpt):
        return torch.sum(inpt * torch.exp(-self.weight)) + torch.sum(self.
            weight)

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)


def get_inputs():
    return [torch.rand([4, 4, 4, 2])]


def get_init_inputs():
    return [[], {}]
