from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.utils.checkpoint


class SeparableConv1D(nn.Module):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, config, input_filters, output_filters, kernel_size,
        **kwargs):
        super().__init__()
        self.depthwise = nn.Conv1d(input_filters, input_filters,
            kernel_size=kernel_size, groups=input_filters, padding=
            kernel_size // 2, bias=False)
        self.pointwise = nn.Conv1d(input_filters, output_filters,
            kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))
        self.depthwise.weight.data.normal_(mean=0.0, std=config.
            initializer_range)
        self.pointwise.weight.data.normal_(mean=0.0, std=config.
            initializer_range)

    def forward(self, hidden_states):
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(initializer_range=4),
        'input_filters': 4, 'output_filters': 4, 'kernel_size': 4}]
