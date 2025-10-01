import torch
from torch import Tensor
from torch import nn


class ScaledConv2d(nn.Conv2d):

    def __init__(self, *args, initial_scale: float=1.0, initial_speed:
        float=1.0, **kwargs):
        super(ScaledConv2d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter('bias_scale', None)
        self._reset_parameters(initial_speed)

    def _reset_parameters(self, initial_speed: 'float'):
        std = 0.1 / initial_speed
        a = 3 ** 0.5 * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        return None if self.bias is None else self.bias * self.bias_scale.exp()

    def _conv_forward(self, input, weight):
        F = torch.nn.functional
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self.
                _reversed_padding_repeated_twice, mode=self.padding_mode),
                weight, self.get_bias(), self.stride, _pair(0), self.
                dilation, self.groups)
        return F.conv2d(input, weight, self.get_bias(), self.stride, self.
            padding, self.dilation, self.groups)

    def forward(self, input: 'Tensor') ->Tensor:
        return self._conv_forward(input, self.get_weight())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
