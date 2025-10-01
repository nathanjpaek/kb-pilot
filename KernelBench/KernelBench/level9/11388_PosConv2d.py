import torch
from torch import Tensor
from torch.utils.data import Dataset as Dataset
import torch.nn.init as init
import torch.utils.data


class PosConv2d(torch.nn.Conv2d):

    def reset_parameters(self) ->None:
        super().reset_parameters()
        self.fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)

    def forward(self, x: 'Tensor') ->Tensor:
        return self._conv_forward(x, torch.nn.functional.softplus(self.
            weight), self.bias) / self.fan_in


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
