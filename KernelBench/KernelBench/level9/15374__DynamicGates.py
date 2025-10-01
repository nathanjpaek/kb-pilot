from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class _DynamicGates(nn.Module):
    """Internal class to wrap the dynamic gate parameters into a dedicated PyTorch Module"""

    def __init__(self, cfg: 'Config', input_size: 'int'):
        super(_DynamicGates, self).__init__()
        self.cfg = cfg
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * cfg
            .hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(cfg.hidden_size, 3 *
            cfg.hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * cfg.hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        nn.init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.cfg.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data
        nn.init.constant_(self.bias.data, val=0)
        if self.cfg.initial_forget_bias is not None:
            self.bias.data[:self.cfg.hidden_size
                ] = self.cfg.initial_forget_bias

    def forward(self, h: 'torch.Tensor', x_d: 'torch.Tensor'):
        gates = h @ self.weight_hh + x_d @ self.weight_ih + self.bias
        return gates


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(hidden_size=4, initial_forget_bias=4),
        'input_size': 4}]
