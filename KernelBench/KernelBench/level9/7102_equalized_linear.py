import torch
import torch.nn as nn
from torch.nn.init import normal
import torch.utils.data


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            'Fan in and fan out can not be computed for tensor with less than 2 dimensions'
            )
    if dimensions == 2:
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


class equalized_linear(nn.Module):

    def __init__(self, c_in, c_out, initializer='kaiming', a=1.0, reshape=False
        ):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':
            normal(self.linear.weight)
        fan_in, _ = _calculate_fan_in_and_fan_out(self.linear.weight)
        gain = (2.0 / (1.0 + a ** 2)) ** 0.5
        self.scale = gain / fan_in ** 0.5
        if reshape:
            c_out /= 4 * 4
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.reshape = reshape

    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        if self.reshape:
            x = x.view(-1, 512, 4, 4)
            x = x + self.bias.view(1, -1, 1, 1).expand_as(x)
        else:
            x = x + self.bias.view(1, -1).expand_as(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c_in': 4, 'c_out': 4}]
