import torch
from torch import Tensor
from torch import nn


class WeightBCE(nn.Module):

    def __init__(self, epsilon: 'float'=1e-08) ->None:
        super(WeightBCE, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: 'Tensor', label: 'Tensor', weight: 'Tensor') ->Tensor:
        """
        :param x: [N, 1]
        :param label: [N, 1]
        :param weight: [N, 1]
        """
        label = label.float()
        cross_entropy = -label * torch.log(x + self.epsilon) - (1 - label
            ) * torch.log(1 - x + self.epsilon)
        return torch.sum(cross_entropy * weight.float()) / 2.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
