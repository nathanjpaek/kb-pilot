import torch
from torch import Tensor
from warnings import warn
from torch.nn import functional as F
from torch.nn import Linear as normal_linear
import torch.utils.data
from torchvision import transforms as transforms


class Linear(normal_linear):

    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        if self.bias is not None:
            warn(
                'A Linear layer has bias, which may be not suitable with weight centralization.'
                )

    def forward(self, input: 'Tensor') ->Tensor:
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        return F.linear(input, weight, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
